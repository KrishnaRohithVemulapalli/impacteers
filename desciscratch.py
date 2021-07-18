import numpy as np
import  pandas as pd
col=['sepal_length','sepal_width','petal_length','petal_width','type']
data = pd.read_csv('iris.data',header=0,names=col)
# print(data)
class Node():
    def __init__(self,feature_index=None,threshold=None,left=None,right=None,info_gain=None,value=None):
        self.feature_index=feature_index    #feature of condition and threshold value of condition
        self.threshold=threshold
        self.left=left
        self.right=right
        self.info_gain=info_gain
        #for leaf node only value
        self.value=value


class decisiontreeClassifier():
    def __init__(self,min_sample_split=2,max_depth=2):
        self.root=None
        self.min_sample_split=min_sample_split
        self.max_depth=max_depth


    def build_tree(self,dataset,curr_depth=0):
        x,y=dataset[:,:-1],dataset[:,-1]
        num_samples,num_features=np.shape(data)
        num_features-=1#150 samples and 4 features
        if num_samples >= self.min_sample_split and curr_depth <= self.max_depth:
            best_split=self.get_best_split(dataset,num_samples,num_features)
            # print(best_split.keys())
            if(best_split['info_gain']>0 ):
                left_subtree=self.build_tree(best_split['dataset_left'],curr_depth+1)
                right_subtree=self.build_tree(best_split['dataset_right'],curr_depth+1)
                return Node(best_split["feature_index"], best_split["threshold"],left_subtree, right_subtree, best_split["info_gain"])

        leaf_value = self.calculate_leaf_value(y)
        # return leaf node
        return Node(value=leaf_value)

    def get_best_split(self, dataset, num_samples, num_features):
        best_split={"feature_index":None,"threshold":None,"dataset_left":None,"info_gain":0}
        max_info_gain=-float("inf")
        for featureindex in range(num_features):
            feature_values=dataset[:,featureindex]   #verticle row of particular feature
            possible_threshold=np.unique(feature_values)
            for threshold in possible_threshold:
                dataset_left,dataset_right=self.split(dataset,featureindex,threshold)
                if(len(dataset_left)>0 and len(dataset_right)>0):
                    y,y_left,y_right=dataset[:,-1],dataset_left[:,-1],dataset_right[:,-1]
                    current_info_gain=self.information_gain(y,y_left,y_right,"gini")
                    if current_info_gain>max_info_gain:
                        best_split["feature_index"] = featureindex
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = current_info_gain
                        max_info_gain = current_info_gain
        return best_split

    def split(self,dataset,featureindex,threshold):
        dataset_left=np.array([row for row in dataset if row[featureindex]<=threshold])
        dataset_right=np.array([row for row in dataset if row[featureindex]>threshold])
        return dataset_left,dataset_right

    def information_gain(self,y,y_left,y_right,mode="entrophy"):
        weight_l=len(y_left)/len(y)
        weight_r=len(y_right)/len(y)
        if(mode=="gini"):
            gain=self.giniindex(y)-(weight_l*self.giniindex(y_left)+weight_r*self.giniindex(y_right))
        else:
            gain = self.entrophy(y) - (weight_l*self.entrophy(y_left)+weight_r*self.entrophy(y_right))
        return gain

    def entrophy(self,y):
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy

    def giniindex(self,y):
        class_labels=np.unique(y)
        gini=0
        for cls in class_labels:
            p_cls=len(y[y==cls])/len(y)
            gini+=p_cls**2
        return 1-gini
    def calculate_leaf_value(self,y):
        Y = list(y)
        return max(Y, key=Y.count)
    ###
    def fit(self, X, Y):
        ''' function to train the tree '''

        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)

    def predict(self, X):
        ''' function to predict new dataset '''

        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions

    def make_prediction(self, x, tree):
        ''' function to predict a single data point '''

        if tree.value != None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)


X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1,1)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=.9, random_state=41)


classifier = decisiontreeClassifier(min_sample_split=3, max_depth=9)
classifier.fit(X_train,Y_train)
Y_pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score

def solve_des(x1,x2,x3,x4):
    output=classifier.predict([[x1,x2,x3,x4]])

    return accuracy_score(Y_test,Y_pred),''.join(output)





