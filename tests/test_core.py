import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from aa3di_benchmark.data import read_fasta
from aa3di_benchmark.runtime import Run
from aa3di_benchmark.statistics import translation_metrics, paired_bootstrap
from aa3di_benchmark.homology import score

ROOT=Path(__file__).resolve().parents[1]


class CoreTests(unittest.TestCase):
    def test_metrics_and_coverage(self):
        result,_=translation_metrics({'a':'AAAC'},{'a':'AAAA'},'AC')
        self.assertEqual(result['micro_accuracy'],.75)
        self.assertAlmostEqual(result['macro_f1'],3/7)
        result,_=translation_metrics({'a':'AA','b':'CC','c':'AA'},{'a':'AA','b':'C'},'ACD')
        self.assertEqual(result['coverage'],1/3)
        self.assertEqual(result['macro_f1'],.5)
        self.assertEqual(result['missing'],1)
        result,_=translation_metrics({'a':'AX'},{'a':'AC'},'AC')
        self.assertEqual(result['unknown_reference_residues'],1)
        result,_=translation_metrics({'a':'aA'},{'a':'AA'},'AC',case_sensitive=True)
        self.assertEqual(result['micro_accuracy'],.5)
        self.assertEqual(result['micro_accuracy'],.5)
        result,_=translation_metrics({'a':'AZ'},{'a':'AA'},'AC',reference_unknowns='XBUZO')
        self.assertEqual(result['unknown_reference_residues'],1)
        with self.assertRaises(ValueError):translation_metrics({'a':'AA'},{'extra':'AA'},'AC')

    def test_pairing(self):
        result=paired_bootstrap({'a':.3,'b':.7},{'b':.5,'a':.1})
        self.assertAlmostEqual(result['mean_difference'],.2)
        self.assertAlmostEqual(result['ci95'][0],.2)
        with self.assertRaises(ValueError):paired_bootstrap({'a':1},{'b':1})

    def test_fasta_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad.fasta';path.write_text('>q\nAA\n>q\nCC\n')
            with self.assertRaises(ValueError):read_fasta(path)

    def test_catalog_first_wrong_fold(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix=Path(tmp)/'catalog'
            score(ROOT/'tests/fixtures/labels.tsv',ROOT/'tests/fixtures/hits.tsv',prefix,8,4)
            self.assertEqual((Path(str(prefix)+'_fam.tsv')).read_text().splitlines()[0],'q1\t1\t1\t1.000000')
            self.assertEqual((Path(str(prefix)+'_sup.tsv')).read_text().splitlines()[0],'q1\t1\t3\t0.333333')
            self.assertEqual((Path(str(prefix)+'_fol.tsv')).read_text().splitlines()[0],'q1\t1\t3\t0.333333')

    def test_run_refuses_changed_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'input';path.write_text('first')
            run=Run(ROOT/'configs/smoke.yaml',Path(tmp)/'out')
            run.record('test','completed',[path]);path.write_text('changed')
            with self.assertRaises(ValueError):run.record('test','running',[path])

    def test_model_identity_across_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'a';b=Path(tmp)/'b';a.write_text('first');b.write_text('second')
            run=Run(ROOT/'configs/smoke.yaml',Path(tmp)/'out')
            run.record('forward','completed',[a],bindings={'model/prostt5':[a]})
            with self.assertRaises(ValueError):
                run.record('reverse','running',[b],bindings={'model/prostt5':[b]})

    def test_clean_cli_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([sys.executable,'-m','aa3di_benchmark.cli','--config',str(ROOT/'configs/smoke.yaml'),
                            '--output',tmp,'smoke'],check=True,cwd=tmp)
            result=json.loads((Path(tmp)/'reports/results.json').read_text())
            self.assertFalse(result['paper_ready'])
            self.assertEqual(result['translation'][0]['micro_accuracy'],.875)
            self.assertTrue((Path(tmp)/'reports/remote_sensitivity.svg').is_file())


if __name__=='__main__':unittest.main()
