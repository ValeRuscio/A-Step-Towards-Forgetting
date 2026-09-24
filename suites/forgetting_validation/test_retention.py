import math
from analyze_retention import functional,analyze
r=functional([dict(q0=.9,margin=math.log(9),support=-math.log(.8)),dict(q0=.1,margin=-math.log(9),support=-math.log(.8))])
assert abs(r['mean_majority_token_probability']-.72)<1e-12
assert r['two_answer_accuracy']==1
assert functional([dict(q0=.9,margin=0,support=0)])['two_answer_accuracy']==.5
rows=[]
losses=[0,1,0,0,0]
for t in range(1,len(losses)):
 before,after=losses[t-1:t+1]
 rows.append(dict(seed=1,step=t,pre={'loss':{'A_test':{'mean':before,'entities':[dict(entity=0,loss=before)]}}},post_losses={'A_test':{'mean':after,'rows':[dict(entity=0,loss=after,q0=.9,margin=1,support=0)]}},functional_consequence={'A_test':{'linear':0,'finite_remainder':after-before},'B_test':{'actual_loss_change':0}}))
a,_=analyze(rows,horizons=(1,4),sustain=2)
assert a[0]['first_sustained_recovery_lag']==1
assert not a[0]['available_h4']
assert a[1]['recovery_status']=='not_applicable'
print('PASS probability reconstruction, tie handling, sustained recovery, follow-up censoring')
