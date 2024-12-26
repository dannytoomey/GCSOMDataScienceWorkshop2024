import pandas as pd
from RelativeRisk import RelativeRisk

data = pd.read_csv('workshop_1/data/data.csv',sep=r'\s*,\s*')

event_e = sum(data['Number_affected_experimental'])
n_e = sum(data['Number_affected_experimental']) + sum(data['Number_not_affected_experimental'])
event_c = sum(data['Number_affected_control'])
n_c = sum(data['Number_affected_control']) + sum(data['Number_not_affected_control'])

risk = RelativeRisk(event_e=event_e,n_e=n_e,event_c=event_c,n_c=n_c)
risk.print_contigency_table()
print("\nRelative risk:",risk.calculate_risk())
ci = risk.calculate_confidence_interval()
print("95% Confidence interval: ",ci[0]," - ",ci[1],"\n")
