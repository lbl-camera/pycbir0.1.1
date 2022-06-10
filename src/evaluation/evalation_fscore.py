import run
import matplotlib.pyplot as plt
import numpy as np
import glob

folders = ['/Users/romuere/Dropbox/CBIR/fibers/database/no_fibers/*','/Users/romuere/Dropbox/CBIR/fibers/database/yes_fibers/*']
fname_database = []
labels_database = np.empty(0)
for id,f in enumerate(folders):
    files = glob.glob(f)
    labels_database = np.append(labels_database, np.zeros(len(files))+id)
    fname_database = fname_database+files
    
feature_extraction_method = 'fotf'
searching_method = 'lsh'
retrieval_number = 2000
similarity_metric = 'ed'
path_output = '/Users/romuere/Dropbox/CBIR/fibers/results/'
list_of_parameters = []
path_cnn_trained = ''



fname_retrieval = fname_database
labels_retrieval = labels_database

result = run.run_command_line(fname_database,labels_database,fname_retrieval,labels_retrieval,path_cnn_trained,path_output,feature_extraction_method,similarity_metric,retrieval_number,list_of_parameters,searching_method, isEvaluation = True)
result = np.array(result[1])
#result = result[1]
fscore = []
#fscore = np.zeros(())
print(len(labels_retrieval))
print(len(fname_database))

for id_label,i in enumerate(labels_retrieval):
    for j in range(retrieval_number): 
        nm = j+1
        nc = sum(result[id_label,:j+1] == i)
        nf = sum(result[id_label,:j+1] != i)
        r = nc/nm
        p = nc/(nc+nf)
        fs = 2*(p*r)/(p+r)
        print(fs)
        fscore.append(fs)
fig = plt.figure()
fig.plot(score)    

fig.show()