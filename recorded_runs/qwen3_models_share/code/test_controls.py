from pathlib import Path
import json,sys,tempfile,time
import controls,open_pilot
with tempfile.TemporaryDirectory() as tmp:
 s=open_pilot.defaults(tmp);s['device']='cpu';s['minimum_free_gib']=0
 controls.start_pilot(s,sys.executable)
 deadline=time.monotonic()+30
 while time.monotonic()<deadline:
  p=Path(tmp)/'status.json';r=json.loads(p.read_text())
  if not controls.alive(tmp) and r['status']!='launching':break
  time.sleep(.1)
 else:raise AssertionError('Worker failed to report status')
 assert r['status']=='dependency_blocked',r
 assert not (Path(tmp)/'assets').exists()
 assert Path(controls.export(tmp)).exists()
 print('PASS detached worker, pre-download native dependency gate, and small report export')
