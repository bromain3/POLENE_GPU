import POLENE_GPU.databaseGeneration.standalone.NeuralCFD as NeuNeu


mech = 'gri30.yaml'
dtb_path = 'sim_symetric_phi_1.0_T_2100_1500_cells.h5'
X, Y = NeuNeu.readH5Database(dtb_path, mech)