import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    assert len(real_labels) == len(predicted_labels)

    true_positive, false_positive, false_negative = 0, 0, 0
    for real, predicted in zip(real_labels, predicted_labels):
        if real == 1 and predicted == 1:
            true_positive += 1
        elif real == 0 and predicted == 1:
            false_positive += 1
        elif real == 1 and predicted == 0 :
            false_negative += 1

    # Check if precision is 0
    if true_positive == 0 and false_positive == 0:
        precision = 0
    else:
        precision = true_positive/float(true_positive + false_positive)
    
    # Check if recall is 0
    if true_positive == 0 and false_negative == 0:
        recall = 0
    else:
        recall = true_positive/float(true_positive + false_negative)

    # Check if both precision and recall are 0
    if precision == 0 and recall == 0:
        return 0

    score = (2*precision*recall)/(precision + recall)
    return score


class Distances:
    @staticmethod
    # TODO
    def canberra_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        res = 0
        for p1, p2 in zip(point1,point2):
            if p1 != 0 and p2 != 0:
                res += (abs(p1 - p2)/(abs(p1) + abs(p2)))
        return res


    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        arr = np.absolute(np.subtract(point1,point2))
        res = np.power(np.sum(np.power(arr,3)), (1/3))
        return res

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        arr = np.subtract(point1,point2)
        res = np.sqrt(np.inner(arr,arr))
        return res

    @staticmethod
    # TODO
    def inner_product_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        res = np.inner(point1,point2)
        return res


    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        res = np.inner(point1,point2)
        norm1 = np.sqrt(np.sum(np.square(point1)))
        norm2 = np.sqrt(np.sum(np.square(point2)))
        res = 1 - (res/(norm1*norm2))
        return res

    @staticmethod
    # TODO
    def gaussian_kernel_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        arr = np.subtract(point1,point2)
        res = -np.exp(-0.5*np.inner(arr,arr))
        return res

class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you should try different distance function you implemented in part 1.1, and find the best k.
        Use k range from 1 to 30 and increment by 2. Use f1-score to compare different models.

        :param distance_funcs: dictionary of distance functions you must use to calculate the distance.
            Make sure you loop over all distance functions for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val:  List[List[int]] Validation data set will be used on your KNN predict function to produce
            predicted labels and tune k and distance function.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_function and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function and self.best_model respectively.
        NOTE: self.best_scaler will be None

        NOTE: When there is a tie, choose model based on the following priorities:
        Then check distance function  [canberra > minkowski > euclidean > gaussian > inner_prod > cosine_dist]
        If they have same distance fuction, choose model which has a less k.
        """

        #best_values = {}
        best_k = 0
        best_score = 0
        best_name = ''
        for name, function in distance_funcs.items():

            for k in range(1, 30, 2):
                # Create model, train, and get f1 score of prediction on new data
                model = KNN(k, function)
                model.train(x_train, y_train)
                accuracy_score = f1_score(y_val, model.predict(x_val))

                # Update best score and associated k
                # K value is increasing, so if there is tie, then the old K value has priority.
                if (accuracy_score > best_score):
                    best_score = accuracy_score
                    best_k = k
                    best_name = name

            # Save best K value for the distance function
            #best_values[name] = [best_score, best_k]
        """
        best_name = ''
        best_score = 0
        best_k = 0

        # Find the best score out of all the attributes
        for i in distance_funcs:
            if best_values[i][0] > best_score:
                best_score = best_values[i][0]
                best_k = best_values[i][1]
                best_name = i                
        """
        # You need to assign the final values to these variables
        self.best_k = best_k
        self.best_distance_function = best_name
        best_model = KNN(best_k, distance_funcs[best_name])
        best_model.train(x_train,y_train)
        self.best_model = best_model
        

    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is similar to Part 1.3 except that before passing your training and validation data to KNN model to
        tune k and disrance function, you need to create the normalized data using these two scalers to transform your
        data, both training and validation. Again, we will use f1-score to compare different models.
        Here we have 3 hyperparameters i.e. k, distance_function and scaler.

        :param distance_funcs: dictionary of distance funtions you use to calculate the distance. Make sure you
            loop over all distance function for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param scaling_classes: dictionary of scalers you will use to normalized your data.
        Refer to test.py file to check the format.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data set you will use on your KNN predict function to produce predicted
            labels and tune your k, distance function and scaler.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_funtion, scaler and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function, self.best_scaler and self.best_model respectively

        NOTE: When there is a tie, choose model based on the following priorities:
        For normalization, [min_max_scale > normalize];
        Then check distance function  [canberra > minkowski > euclidean > gaussian > inner_prod > cosine_dist]
        If they have same distance function, choose model which has a less k.
        """
        # Minmax 
        #scaling0_values = {} 
        # Normalize
        #scaling1_values = {}
        #count = 0
        best_k = 0
        best_score = 0
        best_distance_name = ''
        best_scaler = ''

        for scaling_name, scaling_func in scaling_classes.items():
            # Scale the training and val features according to the scaling functions
            scaler = scaling_func()
            scaled_train_features = scaler(x_train)
            scaled_val_features = scaler(x_val)
            for distance_name, distance_func in distance_funcs.items():

                for k in range(1, 30, 2):
                    # Create model, train, and get f1 score of prediction on new data
                    model = KNN(k, distance_func)
                    model.train(scaled_train_features, y_train)
                    accuracy_score = f1_score(y_val, model.predict(scaled_val_features))

                    # Update best score and associated k
                    # K value is increasing, so if there is tie, then the old K value has priority.
                    if (accuracy_score > best_score):
                        best_score = accuracy_score
                        best_k = k
                        best_distance_name = distance_name
                        best_scaler = scaling_name

                # Save best K value for the distance function and scaling function
                #if count == 0:
                #    scaling0_values[distance_name] = [best_score, best_k, scaling_name]
                #elif count == 1:
                #    scaling1_values[distance_name] = [best_score, best_k, scaling_name]

            # count == 0 for the first scaling function, count == 1 for the second scaling function        
            #count += 1

        #best_distance_name = ''
        #best_scaler = ''
        #best_score = 0
        #best_k = 0

        # Find the best score out of all the attributes
        """
        for i in distance_funcs:
            if scaling0_values[i][0] > best_score:
                # Check minmax
                best_score = scaling0_values[i][0]
                best_k = scaling0_values[i][1]
                best_distance_name = i
                best_scaler = scaling0_values[i][2]
            if scaling1_values[i][0] > best_score:
                # Check normalize
                best_score = scaling1_values[i][0]
                best_k = scaling1_values[i][1]
                best_distance_name = i
                best_scaler = scaling1_values[i][2]
        """
        # You need to assign the final values to these variables
        self.best_k = best_k
        self.best_distance_function = best_distance_name
        self.best_scaler = best_scaler

        best_model = KNN(best_k, distance_funcs[best_distance_name])
        scaler = scaling_classes[best_scaler]()
        best_model.train(scaler(x_train),y_train)
        self.best_model = best_model
        


class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """

        res = []
        for point in features:
            if np.count_nonzero(point) == 0:
                res.append(point)
            else:
                arr = np.sqrt(np.inner(point,point))
                res.append(np.divide(point,arr))
        return res


class MinMaxScaler:
    """
    Please follow this link to know more about min max scaling
    https://en.wikipedia.org/wiki/Feature_scaling
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
    will be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler1 = MinMaxScale()
        train_features_scaled = scaler1(train_features)
        # train_features_scaled should be equal to [[0, 1], [1, 0]]

        test_features_scaled = scaler1(test_features)
        # test_features_scaled should be equal to [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
        self.first_time = 0
        self.minimums = []
        self.maximums = []
        pass
    

    def __call__(self, features):
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        arr = np.array(features)
        if self.first_time == 0:
            self.minimums = np.amin(arr,axis=0)
            self.maximums = np.amax(arr,axis=0)
            self.first_time += 1

        sub = np.subtract(self.maximums,self.minimums)
        sub[0] = 1
        res = []
        for row in features:
            res.append(np.divide(np.subtract(row,self.minimums), sub))
        return res
