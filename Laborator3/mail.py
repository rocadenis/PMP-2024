from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

model = BayesianNetwork([("S", "O"),   
                         ("S", "L"),    
                         ("S", "M"),    
                         ("L", "M")])  

cpd_s = TabularCPD("S", 2, [[0.4], [0.6]])  

cpd_o = TabularCPD("O", 2,
                   [[0.7, 0.1],  
                    [0.3, 0.9]], 
                   evidence=["S"],
                   evidence_card=[2])

cpd_l = TabularCPD("L", 2,
                   [[0.8, 0.3],  
                    [0.2, 0.7]], 
                   evidence=["S"],
                   evidence_card=[2])

cpd_m = TabularCPD("M", 2,
                   [[0.9, 0.6, 0.5, 0.2], 
                    [0.1, 0.4, 0.5, 0.8]], 
                   evidence=["S", "L"],
                   evidence_card=[2, 2])

model.add_cpds(cpd_s, cpd_o, cpd_l, cpd_m)

assert model.check_model()

model.local_independencies(["S", "O", "L", "M"])

inference = VariableElimination(model)

result = inference.query(variables=["S"], evidence={"O": 1, "L": 1, "M": 1})
print(result)
