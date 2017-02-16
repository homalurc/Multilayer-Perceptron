#sample_submission.py
import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(y):
    return y * (1.0 - y)    

class xor_net(object):
    """
    This is a sample class for miniproject 1.
    Args:
        data: Is a tuple, ``(x,y)``
              ``x`` is a two or one dimensional ndarray ordered such that axis 0 is independent 
              data and data is spread along axis 1. If the array had only one dimension, it implies
              that data is 1D.
              ``y`` is a 1D ndarray it will be of the same length as axis 0 or x.   
                          
    """
    def __init__(self, data, labels):
        self.x = data
        self.y = labels       
        self.params = []  # [(w,b),(w,b)] 

        self.x = np.insert(self.x,0,1,axis = 1)
        transpose_x = np.transpose(self.x)    # store transpose of given x
        w_i_1 = np.random.rand(self.x.shape[1],1)   # randomly assign weights to w_i
        w_i_2 = np.random.rand(self.x.shape[1],1)   # randomly assign weights to w_i
        w_i_3 = np.random.rand(self.x.shape[1],1)   # randomly assign weights to w_i        
        learning_rate_1 = 1.0   # initialize learning rate
        learning_rate_2 = 0.001   # initialize learning rate
        learning_rate_3 = 0.001   # initialize learning rate
        alpha_regualrization = 0 # initialize alpha for regularization
        alpha_matrix = alpha_regualrization * np.eye(self.x.shape[1]) 
        count = 5000
        l1_1 = sigmoid(np.dot(self.x, w_i_1))
        l1_2 = sigmoid(np.dot(self.x, w_i_2))
        x_new = np.concatenate(l1_1,l1_2)
        output = sigmoid(np.dot(x_new, w_i_3))

        output_error = (self.y-output)*(1-output)*output

        w_i_3 = w_i_3 + x_new*output_error

        l1_1_error = output_error*w_i_3[0]*l1_1*(1-l1_1)
        l1_2_error = output_error*w_i_3[1]*l1_2*(1-l1_2)

        w_i_1 = w_i_1 + l1_1_error*self.x
        w_i_2 = w_i_2 + l1_2_error*self.x

        while(np.sum(np.absolute(output-self.y)) / float(output.shape[0])<0.1):
            print np.sum(np.absolute(output-self.y)) / float(output.shape[0])

            l1_1 = sigmoid(np.dot(self.x, w_i_1))
            l1_2 = sigmoid(np.dot(self.x, w_i_2))
            x_new = np.concatenate(l1_1,l1_2)
            output = sigmoid(np.dot(x_new, w_i_3))

            output_error = (self.y-output)*(1-output)*output

            w_i_3 = w_i_3 + x_new*output_error

            l1_1_error = output_error*w_i_3[0]*l1_1*(1-l1_1)
            l1_2_error = output_error*w_i_3[1]*l1_2*(1-l1_2)

            w_i_1 = w_i_1 + l1_1_error*self.x
            w_i_2 = w_i_2 + l1_2_error*self.x

        self.params = (w_i_1, w_i_2, w_i_3)
        

        
        




  
    def get_params (self):
        """ 
        Method that should return the model parameters.
        Returns:
            tuple of numpy.ndarray: (w, b). 
        Notes:
            This code will return an empty list for demonstration purposes. A list of tuples of 
            weoghts and bias for each layer. Ordering should from input to outputt
        """
        return self.params

    def get_predictions (self, x):
        """
        Method should return the outputs given unseen data
        Args:
            x: array similar to ``x`` in ``data``. Might be of different size.
        Returns:    
            numpy.ndarray: ``y`` which is a 1D array of predictions of the same length as axis 0 of 
                            ``x`` 
        Notes:
            Temporarily returns random numpy array for demonstration purposes.                            
        """        
        # Here is where you write a code to evaluate the data and produce predictions.
        w_i_1, w_i_2, w_i_3 = self.params

        l1_1 = sigmoid(np.dot(self.x, w_i_1))
        l1_2 = sigmoid(np.dot(self.x, w_i_2))
        x_new = np.concatenate(l1_1,l1_2)
        output = sigmoid(np.dot(x_new, w_i_3))
        print output
        print "hi"
        return output
class mlnn(xor_net):
    """
    At the moment just inheriting the network above. 
    """
    def __init__ (self, data, labels):
        super(mlnn,self).__init__(data, labels)


if __name__ == '__main__':
    pass 