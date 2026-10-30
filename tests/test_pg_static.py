import os
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix

from benchmark.algorithms.pg_exact import PGFilterExact, PGSparseExact
from benchmark.algorithms.pg_dense import PGDenseANN
from benchmark.algorithms.pg_filtered import PGFilteredANN, FUSED_EXACT_PROBE_SQL
from neurips23.streaming.gsvector.module import encode_vector
from benchmark.dataset_io import write_sparse_matrix
from benchmark.datasets import DATASETS
from benchmark.local_dataset import LocalDenseDataset, LocalCSRDataset
from neurips23.ood.gsvector.module import GsvectorOOD


def write_bin(path, array):
    with path.open("wb") as f:
        np.asarray(array.shape, dtype="<u4").tofile(f)
        array.astype("<f4").tofile(f)


class SparseValidationTests(unittest.TestCase):
    def test_negative_and_nonfinite_rejected(self):
        for value in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                PGSparseExact.validate_csr(csr_matrix([[value, 0]]))

    def test_sparse_shape_preserved(self):
        matrix = csr_matrix(([1.0, 2.0], ([0, 1], [1, 30008])), shape=(2, 30109))
        result = PGSparseExact.validate_csr(matrix)
        self.assertEqual(result.shape, (2, 30109))
        self.assertEqual(result.nnz, 2)


class FilterQueryArgumentTests(unittest.TestCase):
    def test_fused_probe_is_explicit_boolean_and_resets(self):
        adapter=object.__new__(PGFilteredANN)
        adapter.set_query_arguments({})
        self.assertFalse(adapter.fused_exact_probe)
        adapter.set_query_arguments({"fused_exact_probe":True})
        self.assertTrue(adapter.fused_exact_probe)
        adapter.set_query_arguments({})
        self.assertFalse(adapter.fused_exact_probe)
        for value in (0,1,"false",None):
            with self.assertRaises(ValueError):
                adapter.set_query_arguments({"fused_exact_probe":value})


@unittest.skipUnless(os.environ.get("GSVECTOR_PG_DSN"), "requires private PostgreSQL")
class StaticIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.environment = patch.dict(os.environ, {
            "GSVECTOR_ARTIFACTS": str(self.root/"events"), "GSVECTOR_KEEP_SCHEMA": "0",
        })
        self.environment.start()
        self.names = []

    def tearDown(self):
        for name in self.names:
            DATASETS.pop(name)
        self.environment.stop()
        self.tmp.cleanup()

    def register(self, name, spec, sparse=False):
        self.names.append(name)
        cls = LocalCSRDataset if sparse else LocalDenseDataset
        spec["groundtruth_dir"] = str(self.root/"gt")
        DATASETS[name] = lambda: cls(name, spec)
        return DATASETS[name]()

    def test_array_cosine_sweeps_and_remote_storage_accounting(self):
        rng = np.random.default_rng(17)
        base = rng.normal(size=(96, 31)).astype("float32")
        adapter = PGDenseANN("angular", {"R":16, "L":64, "build_threads":2,
                                         "min_run_rows":0})
        try:
            adapter.fit(base)
            schema = adapter.schema
            snapshots = []
            backend = adapter.conn.execute("SELECT pg_backend_pid()").fetchone()[0]
            for search_l in (100, 200):
                adapter.set_query_arguments(search_l, 2)
                with patch("psycopg.connect",side_effect=AssertionError("single query opened a new backend")):
                    found = adapter.query(base[7], 10)
                self.assertEqual(found.shape, (10,))
                self.assertEqual(found[0], 7)
                self.assertEqual(adapter.conn.execute("SELECT pg_backend_pid()").fetchone()[0],backend)
                adapter.batch_query(base[:3], 10)
                np.testing.assert_array_equal(adapter.get_batch_results()[:,0],np.arange(3))
                self.assertEqual(adapter.schema, schema)
                snapshots.append(adapter.get_additional())
            for snapshot in snapshots:
                self.assertEqual(hashlib.sha256(Path(snapshot["pg_events_file"]).read_bytes()).hexdigest(),
                                 snapshot["pg_events_sha256"])
            storage = adapter.get_storage_usage()
            self.assertGreater(storage["external_index_bytes"], 0)
            self.assertGreater(storage["internal_relations_bytes"], 0)
            self.assertEqual(storage["index_total_bytes"],
                             storage["external_index_bytes"]+storage["internal_relations_bytes"]
                             +storage["pg_index_bytes"])
            attached = PGDenseANN("cosine", {})
            try:
                attached.attach(schema)
                self.assertEqual(attached.query(base[7],10)[0],7)
                self.assertEqual(attached.live_count,len(base))
            finally:
                attached.done()
            self.assertEqual(adapter.conn.execute("SELECT count(*) FROM base").fetchone()[0],len(base))
            mismatched = PGDenseANN("ip", {})
            try:
                with self.assertRaises(ValueError):
                    mismatched.attach(schema)
            finally:
                mismatched.done()
        finally:
            adapter.done()

    def test_native_ip_and_query_arg_sweep(self):
        base = np.zeros((96, 200), dtype="float32")
        base[:, 0] = np.arange(1, 97)
        queries = np.zeros((2, 200), dtype="float32")
        queries[:, 0] = [1, -1]
        write_bin(self.root/"base.bin", base)
        write_bin(self.root/"queries.bin", queries)
        name = "_test_pg_native_ip"
        self.register(name, {"base": str(self.root/"base.bin"), "queries": str(self.root/"queries.bin"),
                             "metric": "ip"})
        adapter = GsvectorOOD("ip", {"R":16, "L":64, "min_run_rows":0, "build_threads":2})
        try:
            adapter.fit(name)
            schema = adapter.schema
            for threads in (1, 2):
                adapter.set_query_arguments({"search_l":200, "threads":threads})
                adapter.query(queries, 10)
                np.testing.assert_array_equal(adapter.get_results()[:, 0], [95, 0])
                self.assertEqual(adapter.schema, schema)
                self.assertEqual(adapter.conn.execute("SELECT count(*) FROM base").fetchone()[0], 96)
        finally:
            adapter.done()

    def test_random_io_arguments_restore_connection_defaults(self):
        data=np.random.default_rng(913).normal(size=(96,16)).astype("float32")
        adapter=PGDenseANN("euclidean",{"R":16,"L":64,"build_threads":2,"min_run_rows":0})
        try:
            adapter.fit(data)
            initial=adapter.conn.execute("SHOW gsvector.query_random_io").fetchone()[0]
            adapter.set_query_arguments({"search_l":200,"threads":2})
            expected=adapter.query(data[0],10).copy()
            for value in (True,False,None):
                args={"search_l":200,"threads":2}
                if value is not None:args["random_io"]=value
                adapter.set_query_arguments(args)
                np.testing.assert_array_equal(adapter.query(data[0],10),expected)
                current=adapter.conn.execute("SHOW gsvector.query_random_io").fetchone()[0]
                self.assertEqual(current,initial if value is None else "on" if value else "off")
                adapter.batch_query(data[:4],10)
                np.testing.assert_array_equal(adapter.get_batch_results()[:,0],np.arange(4))
            with self.assertRaises(ValueError):
                adapter.set_query_arguments({"random_io":"off"})
            # Single-query session setup caching must notice budget/grouping
            # changes too, not silently keep the old backend GUC values.
            defaults = {name: adapter.conn.execute("SHOW gsvector."+name).fetchone()[0]
                        for name in ("federate_k", "grouped_rerank")}
            for budget, grouped in ((2, True), (3, False), (None, None)):
                args = {"search_l": 200, "threads": 2, "reuse_connections": True}
                if budget is not None:
                    args.update(federate_k=budget, grouped_rerank=grouped)
                adapter.set_query_arguments(args)
                adapter.query(data[0], 10)
                adapter.batch_query(data[:4], 10)
                for conn in [adapter.conn, *adapter._query_connections]:
                    self.assertEqual(float(conn.execute("SHOW gsvector.federate_k").fetchone()[0]),
                                     float(defaults["federate_k"]) if budget is None else budget)
                    self.assertEqual(conn.execute("SHOW gsvector.grouped_rerank").fetchone()[0],
                                     defaults["grouped_rerank"] if grouped is None else
                                     "on" if grouped else "off")
        finally:
            adapter.done()

    def test_native_l1_is_not_l2(self):
        base=np.full((96,8),100,dtype=np.float32)
        base[:,0]+=np.arange(96)
        base[0]=[3,0,0,0,0,0,0,0]
        base[1]=[2,2,0,0,0,0,0,0]
        adapter=PGDenseANN("manhattan",{"R":16,"L":64,"build_threads":2,"min_run_rows":0})
        try:
            adapter.fit(base)
            adapter.set_query_arguments(200,1)
            found=adapter.query(np.zeros(8,dtype=np.float32),10)
            self.assertEqual(found[0],0)  # L2 would prefer row 1.
            expected=np.argsort(np.sum(np.abs(base.astype(np.float64)),axis=1),kind="stable")[:10]
            np.testing.assert_array_equal(found,expected)
        finally:
            adapter.done()

    def test_label_and_empty_and_missing_filters(self):
        base = np.repeat(np.arange(8, dtype="float32")[:, None], 192, axis=1)
        queries = np.zeros((4, 192), dtype="float32")
        write_bin(self.root/"base.bin", base)
        write_bin(self.root/"queries.bin", queries)
        labels = [[1,2], [1], [2], [1,2], [3], [], [1], [2]]
        query_labels = [[1], [1,2], [0], []]
        for name, rows in [("base.meta", labels), ("query.meta", query_labels)]:
            rr, cc = [], []
            for i, terms in enumerate(rows):
                rr.extend([i]*len(terms)); cc.extend(terms)
            write_sparse_matrix(csr_matrix((np.ones(len(rr)), (rr,cc)), shape=(len(rows),4)),
                                str(self.root/name))
        name = "_test_pg_filter"
        ds = self.register(name, {"base": str(self.root/"base.bin"), "queries": str(self.root/"queries.bin"),
                                 "base_metadata": str(self.root/"base.meta"),
                                 "query_metadata": str(self.root/"query.meta")})
        adapter = PGFilterExact("euclidean", {})
        try:
            adapter.fit(name)
            adapter.filtered_query(queries, ds.get_queries_metadata(), 3)
            np.testing.assert_array_equal(adapter.get_results(),
                                          [[0,1,3], [0,3,-1], [-1,-1,-1], [0,1,2]])
        finally:
            adapter.done()

    def test_adaptive_filtered_ann_expansion_and_explicit_fallback(self):
        self._check_adaptive_filtered_ann({})

    def test_filtered_optional_build_strategy_and_layout(self):
        self._check_adaptive_filtered_ann({"build_beam_first":True,"build_neighbor_codes":False})

    def _check_adaptive_filtered_ann(self,build_options):
        base=np.zeros((96,32),dtype=np.float32)
        base[:,0]=np.arange(96)
        queries=base[[0,7,0,0]]
        write_bin(self.root/"base.bin",base)
        write_bin(self.root/"queries.bin",queries)
        labels=[[0]+([1] if i>=80 else [])+([2] if i%2==0 else []) for i in range(96)]
        query_labels=[[1,2],[],[99],[1]]
        for filename,rows in [("base.meta",labels),("query.meta",query_labels)]:
            rr,cc=[],[]
            for i,terms in enumerate(rows):
                rr.extend([i]*len(terms));cc.extend(terms)
            write_sparse_matrix(csr_matrix((np.ones(len(rr)),(rr,cc)),shape=(len(rows),100)),
                                str(self.root/filename))
        name="_test_pg_filter_ann"
        ds=self.register(name,{"base":str(self.root/"base.bin"),"queries":str(self.root/"queries.bin"),
                               "base_metadata":str(self.root/"base.meta"),
                               "query_metadata":str(self.root/"query.meta")})
        adapter=PGFilteredANN("euclidean",{"R":16,"L":64,"build_threads":2,"min_run_rows":0,
                                          **build_options})
        try:
            adapter.fit(name)
            for option,value in build_options.items():
                self.assertEqual(adapter.conn.execute("SHOW gsvector."+option).fetchone()[0],
                                 "on" if value else "off")
            schema=adapter.schema
            snapshots=[]
            for ceiling in (96,8):
                adapter.set_query_arguments({"threads":2,"search_l":16,"initial_candidates":8,
                                             "max_candidates":ceiling,"exact_cutoff":0})
                adapter.filtered_query(queries,ds.get_queries_metadata(),5)
                found=adapter.get_results()
                np.testing.assert_array_equal(found[0],[80,82,84,86,88])
                self.assertEqual(found[1,0],7)
                np.testing.assert_array_equal(found[2],[-1]*5)
                np.testing.assert_array_equal(found[3],[80,81,82,83,84])
                self.assertTrue(adapter.stats["ann_plan_verified"])
                self.assertGreater(adapter.stats["ann_queries"],0)
                self.assertEqual(adapter.schema,schema)
                if ceiling==8:
                    self.assertGreaterEqual(adapter.stats["fallback_exact_queries"],2)
                else:
                    self.assertGreater(adapter.stats["ann_attempts"],adapter.stats["ann_queries"])
                snapshots.append(adapter.get_additional())
            for item in snapshots:
                self.assertEqual(hashlib.sha256(Path(item["pg_events_file"]).read_bytes()).hexdigest(),
                                 item["pg_events_sha256"])
                self.assertTrue(item["native_ann"])
            # Compare identical routing/budgets; include small, empty, broad,
            # conjunction and unfiltered predicates, and fewer-than-k results.
            for cutoff in (8,96):
                outputs=[]
                route_counts=[]
                for fused in (False,True):
                    adapter.set_query_arguments({
                        "threads":2,"search_l":16,"initial_candidates":8,
                        "max_candidates":96,"exact_cutoff":cutoff,
                        "fused_exact_probe":fused})
                    adapter.filtered_query(queries,ds.get_queries_metadata(),5)
                    outputs.append(adapter.get_results().copy())
                    route_counts.append(tuple(adapter.stats[key] for key in
                        ("small_exact_queries","ann_queries","fallback_exact_queries","ann_attempts")))
                    self.assertEqual(adapter.stats["fused_probe_queries"],3 if fused else 0)
                    for phase in ("probe_s","small_exact_s","ann_s","fallback_exact_s"):
                        self.assertGreaterEqual(adapter.stats[phase],0)
                np.testing.assert_array_equal(*outputs)
                self.assertEqual(*route_counts)
            # A full bounded set is exact; cutoff+1 must return NO partial top-k.
            with adapter._connect() as conn:
                for cutoff,expected in ((15,[]),(16,[80,81,82,83,84])):
                    count,found=conn.execute(FUSED_EXACT_PROBE_SQL,
                        ([1],cutoff+1,cutoff,encode_vector(queries[0]),5)).fetchone()
                    self.assertEqual(count,16)
                    self.assertEqual(found,expected)
                count,found=conn.execute(FUSED_EXACT_PROBE_SQL,
                    ([1,2],9,8,encode_vector(queries[0]),10)).fetchone()
                self.assertEqual(count,8)
                self.assertEqual(found,list(range(80,96,2)))
                # UPDATE creates new tuple versions: CTIDs must be rediscovered,
                # and equal distances must still use the exact path's ID tie-break.
                conn.execute("UPDATE base SET embedding=%s WHERE id>=80",
                             (encode_vector(queries[0]),))
                count,found=conn.execute(FUSED_EXACT_PROBE_SQL,
                    ([1],17,16,encode_vector(queries[0]),5)).fetchone()
                self.assertEqual(count,16)
                self.assertEqual(found,[80,81,82,83,84])
        finally:
            adapter.done()

    def test_sparse_high_terms_and_zero_score_fill(self):
        base = csr_matrix(([2,4,5,1,1], ([0,0,1,2,2], [1,30008,2,1,2])),
                          shape=(4,30109), dtype="float32")
        queries = csr_matrix(([1,1,1,1], ([0,1,1,3], [30008,1,2,3])),
                             shape=(4,30109), dtype="float32")
        write_sparse_matrix(base, str(self.root/"base.csr"))
        write_sparse_matrix(queries, str(self.root/"query.csr"))
        name = "_test_pg_sparse"
        self.register(name, {"base": str(self.root/"base.csr"), "queries": str(self.root/"query.csr")},
                      sparse=True)
        adapter = PGSparseExact("ip", {})
        try:
            adapter.fit(name)
            adapter.query(queries, 3)
            np.testing.assert_array_equal(adapter.get_results(),
                                          [[0,1,2], [1,0,2], [0,1,2], [0,1,2]])
        finally:
            adapter.done()


if __name__ == "__main__":
    unittest.main()
