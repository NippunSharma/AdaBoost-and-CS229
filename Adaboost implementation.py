from sklearn.ensemble import RandomForestClassifier
import numpy as np

class Adaboost:

    def __init__(self,n_base_learners=6):
        self.base_learners=n_base_learners
        self.B=None
        self.D=None
        self.h_fin,self.beta=None,None
        self.n_labels=None
    
    def fit(self,X,y,verbose=0):
        n_samples,n_features = X.shape
        n_labels = len(np.unique(y))
        self.n_labels=n_labels
        
        # mislabel pairs 
        self.B=[]
        
        for i in range(n_samples):
            l=[]
            for j in range(0,n_labels):
                if(j!=y[i]):
                    l.append([i,j])
            self.B.extend(l)
        
        n_mislabel_pairs=(n_samples)*(n_labels-1)
        
        # initial distribution
        self.D=np.zeros((n_samples,n_labels))
        self.D.fill(1/n_mislabel_pairs)
        
        
        h_fin=[]
        beta=[]
        
        # main loop
        for t in range(self.base_learners):
            if(verbose==1):
                print('Training base learner:',t+1)
            h=[]
            
            # Weak Learner -- Random Forest (Sklearn) Can change parameters to make single decision tree or decision stump
            clf=RandomForestClassifier(n_estimators=3,max_depth=1)
            clf.fit(X,y)
            proba=clf.predict_proba(X)
            
            for i in range(n_samples):
                l=[]
                for j in range(1,n_labels +1):
                    l.append(proba[i][j-1])
                h.append(l)
        
            h=np.array(h)
            
            # calculating pseudo loss
            pseudo_loss = 0
            
            for i,j in self.B:
                pseudo_loss+=self.D[i,j]*(1-h[i,y[i]]+h[i,j])
            pseudo_loss=0.5*pseudo_loss
        
            if(abs(pseudo_loss)<1e-10):
                pseudo_loss=1e-10
            
            # calculating beta
            beta_t=pseudo_loss/(1-pseudo_loss)
            
            # updating ditribution and finding normalization constant
            Z=0
            for i in range(n_samples):
                for j in range(n_labels):
                    self.D[i,j]=self.D[i,j]*(beta_t)**(0.5*(1 + h[i,y[i]] - h[i,j]))
                    Z+=self.D[i,j]
            self.D=self.D/Z
            beta.append(beta_t)
            h_fin.append(clf)
            if(verbose==1):
                print('Weight of learner:',np.math.log(1/beta_t),'\n')
            
        self.h_fin=h_fin
        self.beta=beta
    
    def predict(self,X):
        
        h_fin=self.h_fin
        beta=self.beta
    
        output_list=[]
        for j in range(self.n_labels):
            l=0
            for t in range(self.base_learners):
                proba=h_fin[t].predict_proba(X.reshape(1,-1))
                l+=np.math.log(1/beta[t])*(proba[0][j])
            output_list.append(l)
        max_val=max(output_list)
        
        return output_list.index(max_val)



# testing the algorithm  
        
from sklearn import datasets
from sklearn.model_selection import train_test_split

data = datasets.load_wine()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
clf=Adaboost()

clf.fit(X_train,y_train,verbose=1)
def accuracy(X,y):
    n_samples=X.shape[0]
    preds=[]
    for i in range(n_samples):
        pred=clf.predict(X[i])
        preds.append(pred)
    y=np.array(y)
    preds=np.array(preds)
    accuracy = np.sum(y == preds)/len(y)
    return accuracy,preds
accuracy,preds=accuracy(X_test,y_test)
print('Accuracy:',accuracy)







