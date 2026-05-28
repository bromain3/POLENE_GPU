import matplotlib.pyplot as plt

def plotDistrib(X, colName):
    plt.hist(X[colName].cpu().detach().numpy(), bins=100, alpha=0.5, color='blue', rwidth=0.6)
    plt.xlabel(colName)
    plt.ylabel('Sample number')
    plt.show()

def plotAllDistrib(X):
    for col in X.colName:
        plt.hist(X[col].cpu().detach().numpy(), bins=100, alpha=0.5, color='blue', rwidth=0.6)
        plt.xlabel(col)
        plt.ylabel('Sample number')
        plt.show()
