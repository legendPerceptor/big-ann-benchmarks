"""Adaptive native L2 ANN + GIN exact fallback for conjunction label filters.

The WHERE clause is evaluated inside PostgreSQL for every ANN candidate.
Insufficient candidates trigger bounded expansion, then an explicitly counted
exact fallback. This is not an inline graph-label filter and does not claim one.
"""
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from psycopg import sql

from benchmark.algorithms.pg_exact import PGFilterExact
from benchmark.algorithms.pg_dense import PGDenseANN
from neurips23.streaming.gsvector.module import encode_vector, has_index_plan


# Keep only snapshot-local CTIDs, not full vectors, in the bounded tuplestore.
# A TID scan reuses the probe's heap pages without rebuilding the GIN bitmap.
# Broad predicates do no distance work; CTIDs never escape this statement.
# The +0 sort expression prevents an approximate ordered index scan even if a
# caller enables indexscan (an ID tie-break alone can use incremental sort).
# The guard must be evaluated BEFORE LIMIT k: a truncated eligible set must
# never be returned as an exact answer. All work uses the query's MVCC snapshot.
FUSED_EXACT_PROBE_SQL = """
    WITH candidates AS MATERIALIZED (
        SELECT ctid AS tid
        FROM base WHERE tags @> %s::integer[] LIMIT %s
    ), cardinality AS (
        SELECT count(*) AS n FROM candidates
    )
    SELECT n, ARRAY(
        SELECT id FROM base
        WHERE cardinality.n <= %s
          AND ctid = ANY(ARRAY(SELECT tid FROM candidates))
        ORDER BY (embedding <-> %s) + 0, id LIMIT %s
    ) FROM cardinality
"""


class PGFilteredANN(PGFilterExact):
    mode = "gsvector-filtered-ann-with-gin-exact-fallback"

    def __init__(self,metric,index_params):
        params=dict(index_params)
        allowed={"batch_rows","R","L","alpha","build_threads","min_run_rows",
                 "build_beam_first","build_neighbor_codes"}
        if set(params)-allowed:
            raise ValueError("unsupported filtered ANN build parameters")
        super().__init__(metric,{"batch_rows":params.get("batch_rows",4096)})
        self.R=int(params.get("R",32))
        self.L=int(params.get("L",100))
        self.alpha=float(params.get("alpha",1.2))
        self.build_threads=int(params.get("build_threads",4))
        self.min_run_rows=int(params.get("min_run_rows",32768))
        self.build_options={name:params[name] for name in ("build_beam_first","build_neighbor_codes")
                            if name in params}
        if any(not isinstance(value,bool) for value in self.build_options.values()):
            raise ValueError("build strategy/layout options must be boolean")
        if (not 1<=self.R<=self.L<=4096 or not 1<=self.alpha<=4 or
                self.build_threads<1 or self.min_run_rows<0):
            raise ValueError("invalid filtered ANN build parameters")
        self.query_number=0
        self.stats={}
        self.set_query_arguments({})

    def fit(self,dataset):
        start=time.perf_counter()
        super().fit(dataset)
        loaded=time.perf_counter()
        self.live_count=self.nb
        self.conn.execute("SELECT set_config('gsvector.build_threads',%s,false)",
                          (str(self.build_threads),))
        self.conn.execute("SELECT set_config('gsvector.min_run_rows',%s,false)",
                          (str(self.min_run_rows),))
        for name,value in self.build_options.items():
            self.conn.execute("SELECT set_config(%s,%s,false)",("gsvector."+name,"on" if value else "off"))
        self.conn.execute(sql.SQL(
            "CREATE INDEX ann_idx ON base USING gsvector(embedding vector_l2_ops) "
            "WITH(engine='diskann',r={},l={},alpha={})"
        ).format(sql.Literal(self.R),sql.Literal(self.L),sql.Literal(self.alpha)))
        self.build_times={"load_and_gin_s":loaded-start,
                          "ann_index_s":time.perf_counter()-loaded}
        self._record("ann_fit",rows=self.nb,parameters={"R":self.R,"L":self.L,"alpha":self.alpha,
                                                     **self.build_options},
                     build_times=self.build_times)

    def set_query_arguments(self,args):
        allowed={"threads","search_l","initial_candidates","max_candidates","exact_cutoff",
                 "fused_exact_probe"}
        if set(args)-allowed:
            raise ValueError("unsupported filtered ANN query parameters")
        self.threads=int(args.get("threads",4))
        self.search_l=int(args.get("search_l",200))
        self.initial_candidates=int(args.get("initial_candidates",256))
        self.max_candidates=int(args.get("max_candidates",8192))
        self.exact_cutoff=int(args.get("exact_cutoff",2048))
        self.fused_exact_probe=args.get("fused_exact_probe",False)
        if not isinstance(self.fused_exact_probe,bool):
            raise ValueError("fused_exact_probe must be boolean")
        if (self.threads<1 or not 1<=self.search_l<=32768 or self.exact_cutoff<0 or
                not 1<=self.initial_candidates<=self.max_candidates<=32768):
            raise ValueError("invalid filtered ANN query parameters")

    def filtered_query(self,X,filters,k):
        start=time.perf_counter()
        progress_interval=int(os.environ.get("GSVECTOR_FILTER_PROGRESS_INTERVAL","0"))
        if progress_interval<0:
            raise ValueError("negative filtered query progress interval")
        if (X.shape!=(filters.shape[0],self.dim) or not np.isfinite(X).all() or
                not 1<=k<=self.max_candidates):
            raise ValueError("invalid filtered queries")
        filters=filters.tocsr()
        # At large N, predicates rarer than k/max_candidates are unlikely to
        # fill k slots even at the ANN ceiling. Prefer their bounded GIN set.
        # An explicit cutoff of zero disables this rule for diagnostics.
        cutoff=self.exact_cutoff
        if cutoff:
            cutoff=max(cutoff,min(self.nb,(2*k*self.nb+self.max_candidates-1)//self.max_candidates))
        ann_sql="SELECT id FROM base WHERE tags @> %s::integer[] ORDER BY embedding <-> %s LIMIT %s"
        exact_sql="SELECT id FROM base WHERE tags @> %s::integer[] ORDER BY embedding <-> %s,id LIMIT %s"

        def batch(indices):
            result=np.full((len(indices),k),-1,dtype=np.int32)
            stats={"small_exact_queries":0,"ann_queries":0,"fallback_exact_queries":0,
                   "ann_attempts":0,"max_candidates_used":0,"ann_plan_verified":False,
                   "fused_probe_queries":0,"probe_s":0.0,"small_exact_s":0.0,
                   "ann_s":0.0,"fallback_exact_s":0.0}
            with self._connect() as conn:
                conn.execute("SET gsvector.federate_k=1; SET gsvector.iterate_max=0; "
                             "SET gsvector.query_beam_first=on")
                for row,qi in enumerate(indices):
                    tags=filters.indices[filters.indptr[qi]:filters.indptr[qi+1]].tolist()
                    params=(tags,encode_vector(X[qi]),k)
                    # All expansion attempts and exact fallback share one MVCC
                    # snapshot even if another client updates labels/vectors.
                    with conn.transaction():
                        conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
                        conn.execute("SET LOCAL enable_bitmapscan=on; SET LOCAL enable_indexscan=off")
                        phase_start=time.perf_counter()
                        fused_values=None
                        if tags and cutoff and self.fused_exact_probe:
                            eligible,fused_values=conn.execute(
                                FUSED_EXACT_PROBE_SQL,
                                (tags,cutoff+1,cutoff,params[1],k)).fetchone()
                            stats["fused_probe_queries"]+=1
                        else:
                            eligible=self.nb if not tags else conn.execute("""
                                SELECT count(*) FROM (
                                  SELECT 1 FROM base WHERE tags @> %s::integer[] LIMIT %s
                                ) candidates
                            """,(tags,cutoff+1)).fetchone()[0]
                        stats["probe_s"]+=time.perf_counter()-phase_start
                        if eligible<=cutoff:
                            phase_start=time.perf_counter()
                            values=(fused_values if fused_values is not None else
                                    [r[0] for r in conn.execute(exact_sql,params)])
                            stats["small_exact_s"]+=time.perf_counter()-phase_start
                            stats["small_exact_queries"]+=1
                        else:
                            conn.execute("SET LOCAL enable_indexscan=on; SET LOCAL enable_bitmapscan=off")
                            budget=min(self.nb,max(k,self.initial_candidates))
                            limit=min(self.nb,max(k,self.max_candidates))
                            values=[]
                            while True:
                                conn.execute("SELECT set_config('gsvector.max_k',%s,true),"
                                             "set_config('gsvector.search_l',%s,true)",
                                             (str(budget),str(max(self.search_l,budget))))
                                if not stats["ann_plan_verified"]:
                                    plan=conn.execute("EXPLAIN (FORMAT JSON) "+ann_sql,params).fetchone()[0]
                                    if not has_index_plan(plan,"ann_idx"):
                                        raise RuntimeError("filtered ANN did not select its native index")
                                    stats["ann_plan_verified"]=True
                                phase_start=time.perf_counter()
                                values=[r[0] for r in conn.execute(ann_sql,params)]
                                stats["ann_s"]+=time.perf_counter()-phase_start
                                stats["ann_attempts"]+=1
                                stats["max_candidates_used"]=max(stats["max_candidates_used"],budget)
                                if len(values)==k:
                                    stats["ann_queries"]+=1
                                    break
                                if budget>=limit:
                                    conn.execute("SET LOCAL enable_indexscan=off; SET LOCAL enable_bitmapscan=on")
                                    phase_start=time.perf_counter()
                                    values=[r[0] for r in conn.execute(exact_sql,params)]
                                    stats["fallback_exact_s"]+=time.perf_counter()-phase_start
                                    stats["fallback_exact_queries"]+=1
                                    break
                                budget=min(limit,budget*2)
                        if len(set(values))!=len(values) or any(v<0 or v>=self.nb for v in values):
                            raise RuntimeError("invalid filtered neighbor IDs")
                        result[row,:len(values)]=values
                    if progress_interval and (row+1)%progress_interval==0:
                        self._record("filtered_query_progress",query_number=self.query_number+1,
                                     chunk_start=int(indices[0]),completed=row+1,
                                     chunk_queries=len(indices),elapsed_s=time.perf_counter()-start,
                                     stats=dict(stats))
            return result,stats

        chunks=[x for x in np.array_split(np.arange(len(X)),self.threads) if len(x)]
        with ThreadPoolExecutor(max_workers=self.threads) as pool:
            outputs=list(pool.map(batch,chunks))
        self.res=np.concatenate([x[0] for x in outputs]) if outputs else np.empty((0,k),dtype=np.int32)
        keys=("small_exact_queries","ann_queries","fallback_exact_queries","ann_attempts",
              "fused_probe_queries","probe_s","small_exact_s","ann_s","fallback_exact_s")
        self.stats={key:sum(x[1][key] for x in outputs) for key in keys}
        self.stats["max_candidates_used"]=max((x[1]["max_candidates_used"] for x in outputs),default=0)
        self.stats["ann_plan_verified"]=any(x[1]["ann_plan_verified"] for x in outputs)
        self.stats["effective_exact_cutoff"]=cutoff
        self.stats["fused_exact_probe"]=self.fused_exact_probe
        self.stats["timing_scope"]="summed worker wall seconds; fused exact work is included in probe_s"
        self.query_number+=1
        self._record("filtered_query",queries=len(X),k=k,elapsed_s=time.perf_counter()-start,
                     mode=self.mode,stats=self.stats)

    def get_additional(self):
        snapshot=self.log.with_name(self.log.stem+f"-snapshot-{self.query_number}.jsonl")
        snapshot.write_bytes(self.log.read_bytes())
        storage=PGDenseANN.get_storage_usage(self)
        gin=self.conn.execute("SELECT pg_relation_size(%s::regclass)",(f"{self.schema}.tags_idx",)).fetchone()[0]
        storage["gin_bytes"]=gin
        storage["index_total_bytes"]+=gin
        storage["index_bytes_per_vector"]=storage["index_total_bytes"]/max(1,self.nb)
        return {**super().get_additional(),"native_ann":True,
                "pg_events_file":str(snapshot.resolve()),
                "pg_events_sha256":hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                "pg_filter_stats_json":json.dumps(self.stats),
                "pg_storage_json":json.dumps(storage),
                "filtering_method":"SQL post-filter + bounded expansion + explicit exact fallback"}
