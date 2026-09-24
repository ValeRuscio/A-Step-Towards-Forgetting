"""Separate subprocess test of the actual three-stage launcher, with tiny Qwen3."""
import tempfile,time,sys,json,sqlite3
from pathlib import Path
import replication as r

def main():
    with tempfile.TemporaryDirectory() as td:
        s=r.defaults(td);s.update(python=sys.executable,smoke=True,device='cpu',threads=1,seeds=[61],hours=.5,minimum_free_gib=0,max_output_gib=3)
        r.launch(s)
        for _ in range(900):
            st=r.status(s)
            if st['status'] in ['failed','paused','complete','complete_with_limitations'] and not st['alive']:break
            time.sleep(.2)
        assert st['status'] in ['complete','complete_with_limitations'],st
        p=Path(td)
        with sqlite3.connect(p/'natural/results.sqlite') as c:
            counts=dict(c.execute('SELECT kind,COUNT(*) FROM records GROUP BY kind'));assert counts['natural_done']==2 and counts['deferred_paths_done']==2 and counts['path_done']==4,counts
        hs=r.read(p/'history/summary.json');assert hs['updates']==6 and hs['replay_hashes_pass'] and hs['algebra_pass'],hs
        assert len(r.read(p/'history/cohort_timeline.json'))>0
        pairs=r.read(p/'natural/answer_condition_pairs.json');assert pairs[0]['identical_A_weights'] and pairs[0]['identical_example_selection'],pairs
        assert Path(r.export(s)).exists()
        assert r.launch(s)['status'] in ['complete','complete_with_limitations']
        print('PASS: three stages, tiny Qwen3 paired conditions, exact replay, cohort export, finite paths, idempotent launch.')
if __name__=='__main__':main()
