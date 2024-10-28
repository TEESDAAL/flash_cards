flashcards = {
    "Vectors, matrices, spaces": [
        ("How do we take the dot product of two vectors (v•w)?", "We multiply them element wise v•w=Σv_iw_i ([1,2]•[3,4] = 1*3+2*4 = 11)"),
        ("What is a vector?", "A collection of numbers, can represent a point in space or a direction"),
        ("What does a it mean when the dot product of two vectors = 0?", "The two vectors are orthogonal (at right angles)"),
        ("An (n x m) matrix can be multiplied by a matrix of what shape?", "(m x n)"),
        ("We can use a matrix to project a vector up a dimension (say from 2->3 dim), do we gain any degrees of freedom when we do this?", "no"),
        ("What does SVD (Singular Value Decomposition) do?", "SVD breaks down any matrix M into three simpler matrices, which rotate, scale, and then rotate again"),
        ("What is the main application of SVD that we discussed in lectures?", "We can use it to compress matricies (for example images), by selecting the first k singular values"),
        ("What does the dot product of two vectors equal as the number of dimensions increases?", "0 as in high-dimensional spaces, if you generate random vectors*, the angles between them will almost always be close to 90 degrees"),
        ("Describe the curse of dimensionality:", "As the number of dimensions increases, the volume of the space increases exponentially, meaning that with more dimensions, data points become sparse."),
        ("What tends to happen to the distance between vectors when the dimensionality increases", "The distance between any two points tends to become similar, making it hard to distinguish between near and far points. This affects algorithms that rely on distance measures, like k-means clustering and k-nearest neighbours."),
    ],

    "Linear Regression": [
        ("How do we fit a linear regression model?", "We can:\n 1. optimize the model iteratively using gradient descent.\n 2. Solve in 1 step using a pseudo inverse.\n3. Or we can infer the best solution via probability theory."),
        ("How do we get the predictions from the model, Y_pred = ?", "Y_pred = XW, where X is a matrix holding the inputs to the model, and W is the learned weights of the model"),
        ("What is the gradiant operator (∇), and why do we care about it?", "The gradient operator gives us a vector of the partial derivatives of some function f(𝑥_1, 𝑥_2, …, 𝑥_𝑛) with respect to all it's arguments, This vector ∇_x f  points in the direction of the steepest ascent."),
        ("What is the gradient of a linear function", "Just some constant :)"),
        ("Given f(x) x^Tw, where x and w are both vectors, what is ∇_x f?", "∇_x f = w"),
        ("What is the assumed model for linear regression", "Y = X ⋅ w_true + e_noise"),
        ("How do we calculate an approximation for the true weights for linear model?","Ŵ = (inverse(X^TX) X^T) Y), or the pseudo_inverse(X) Y"),
        ("In reality, we don't use the true pseudo_inverse when calculating the weights, we add a small regularising term W = inverse(λ1 + X^TX) X^T. Why?", "1. (X^TX) might not be invertible. Adding 𝜆1 ensures (X^TX + 𝜆1) is invertible. \n 2. The term 𝜆1 acts as a regularization term. It penalizes large weights, helping prevent overfitting."),
    ],

    "Probabilities": [
        ("""
      i │ ii │ iii
    ┌───┬────┬─────┐
  A │ 2 │  2 │  1  │
  B │ 1 │  4 │  2  │
    └───┴────┴─────┘
What is the P(A,i)?""", "2/12 ≈ 0.1667"),
("""
      i │ ii │ iii
    ┌───┬────┬─────┐
  A │ 2 │  2 │  1  │
  B │ 1 │  4 │  2  │
    └───┴────┴─────┘
What is the P(A,iii)?""", "1/12 ≈ 0.0833"),
("""
      i │ ii │ iii
    ┌───┬────┬─────┐
  A │ 2 │  2 │  1  │
  B │ 1 │  4 │  2  │
    └───┴────┴─────┘
What is the P(B,ii)?""", "4/12 ≈ 0.3333"),
("""
      i │ ii │ iii
    ┌───┬────┬─────┐
  A │ 2 │  2 │  1  │
  B │ 1 │  4 │  2  │
    └───┴────┴─────┘
What is the P(i | A)?""", "2/5 = 0.4"),
("""
      i │ ii │ iii
    ┌───┬────┬─────┐
  A │ 2 │  2 │  1  │
  B │ 1 │  4 │  2  │
    └───┴────┴─────┘
What is the P(A)?""", "2/3 ≈ 0.6667"),
("""
      i │ ii │ iii
    ┌───┬────┬─────┐
  A │ 2 │  2 │  1  │
  B │ 1 │  4 │  2  │
    └───┴────┴─────┘
What is the P(A) + P(B)?""", "1"),
("""
      i │ ii │ iii
    ┌───┬────┬─────┐
  A │ 2 │  2 │  1  │
  B │ 1 │  4 │  2  │
    └───┴────┴─────┘
What is the P(A, B)?""", "0"),
        ("Insert brackets into the following equation in a way that doesn't change the order of operations: P(X, Y | Z)", "P((X, Y) | Z)"),
        ("Insert brackets into the following equation in a way that doesn't change the order of operations: P(X | Y, Z)", "P(X | (Y, Z))"),
        ("Write the product rule for probabilities", "P(X) P(Y|X) = P(X,Y)"),
        ("How to find the probability of P(A|B) given P(A, B), P(B), P(A)", "P(A|B)=P(A, B)/P(B)"),
        ("Find P(Y|X) from P(X) and P(X,Y)", "P(Y|X) = P(X,Y)/P(X)"),
        ("What is the sum rule for probabilities?", "P(X=x) = ∑_i P(X=x, Y=y_i)"),
        ("Given the Joint distribution of X,Y,Z. How do you find P(X=x| Y=y)?", "P(X=x|Y=y) = P(X=x,Y=y)/P(Y=y)\n = (∑_i P(X=x, Y=y, Z=z_i)/∑_i∑_jP(X=x_i, Y=y, Z=z_i)"),
        ("What are the two ways to express independence between X and Y?", "1. P(X,Y) = P(X)*P(Y)\n2.P(X|Y) = P(X)"),
        ("If X and Y are independent given Z what does P(X | Y, Z) equal?", "P(Y|Z)"),
        ("If P(cat) = 0.08, P(dog) = 0.02, P(cow) = 0.9 What is the chance that I see a data set, D = (cow, cow, dog)? (Note the slides make the assumption that these events are independent.", "If the sightings are independent:\nP(D) = P(cow, cow, dog)\n = P(cow)P(cow)P(dog)\n = 0.9*0.9*0.2=0.162"),
        ("How do we calculate the Maximum likelihood estimators for the probabilities of a catgorical distribution?", "Our estimator of P(X=x_i), P̂(X=x_i) = count(x_i)/N, where count(x_i) gives the number of times a value occurs in the dataset, and N is the size of the dataset."),
        ("Why would we not want to directly use the counts of all the classes? If not why and provide an example?", "If the number of entries is small we run into problems with overfitting, consider if we have only seen 1 datapoint, then we have one catagory with 100% probability and everything else has 0%. (To solve this we add 1 to all the counts, called laplace smoothing)"),
        ("What is the formula for entropy?", "H(X) = ∑_i P(X=x_i)log_2(1/P(X=x_i))"),
        ("What does an entropy H(X)=0 actually mean?", "H(X)=0, means that there is no uncertainty, that is all of the data falls under 1 catagory."),
        ("Under what circumstance do we maximise probabilities?", "It means the probabilities are split evenly between the catagories, fun fact this would leave an entropy of log_2(n) where n is the number of catagories.")
    ],
    "Inference and Networks": [
        ("What does it mean to factorize P(X, Y)", "The product rule factorises the joint P(X, Y) into P(X)P(Y|X)."),
        ("What does Bayes rule allow us to do?", "It allows us to infer “backwards”, from effects to cause. Or more abstractly it allows us to find P(B|A) in terms of P(A|B)"),
        ("What is the formula of Bayes rule?", "P(hypothesis ∣ data) = (P(hypothesis) × P(data ∣ hypothesis))/P(data)"),
        ("How do you calculate the Bayes Factor between two hypotheses, H_1, H_2 for some given data D", "The bayes factor is given by: (P(H_1)/P(H_2)) × (P(D|H_1)/P(D|H_2))"),
        ("Calculate P(dog | meow), given that P(dog) = 4/5, P(meow | dog) = 1/10, P(meow | cat) = 19/20.", """
P(dog | meow)/P(cat ∣ meow) = (P(dog)/P(cat)) × (P(meow ∣ dog)/P(meow ∣ cat))
    = ((4/5)/(1/5)) × ((1/10)/(19/20))
    = 4 × (2/19)
    = 8/19

=> P(dog | meow) = 8/(19+8)
    = 8/27
    ≈ 0.2962
"""),
        ("How does P(X, Y, Z) “factorise”", "P(X, Y, Z) = P(X)P(Y|X)P(Z|X,Y)"),
        ("Compare how P(X, Y, Z) factorise if X, Y, Z are indepenent and not", """
Not independent:
P(X, Y, Z) = P(X)P(Y|X)P(Z|X,Y)
Independent:
P(X, Y, Z) = P(X)P(Y)P(Z)

We can see independence dramatically simplifies the factorisation.
"""),
        ("How does P(A, B, C) factorise if the graph motif is a chain (A and C are conditionally independent, given B)", "P(A, B, C) = P(A)P(B|A)P(C|B)"),
        ("How does P(A, B, C) factorise if the graph motif is a branch (B and C are conditionally independent, given A)", "P(A, B, C) = P(A)P(B|A)P(C|A)"),
        ("How does P(A, B, C) factorise if the graph motif is a collider (A and B are already independent, but they become dependent, once you observe C.)",
         "P(A, B, C) = P(A)P(B)P(C|A,B)"),
    ],

    "Gaussians": [
        ("How do you find the mean of a dataset?", "The average of a dataset X (which is a matrix) is given be the vector which has entries μ_i =  1/n ∑_jX_i,j\n Or in python μ⃗ = [sum(row)/len(row) for row in X]"),
        ("How do you find the covariance of a dataset?", "Cov = 1/n∑_i (X_i - μ)(X_i - μ)^T = "),
        ("What does it mean to standardize a dataset?", "Standardization is centering and then scaling the individual feature variances to be 1. Standardised data will have 𝜇⃗ = 0⃗ and the diagonals of Covariance Matrix=1."),
        ("Briefly describe the central limit theorem", "Sums of independent random variables from almost any distribution approach a Normal distribution"),
        ("What is the notation for the standard normal distribution", "N(0,1) = 1/sqrt(2pi)e^(-x^2/2)"),
        ("What is the formula for the parameterized normal distribution?", "N(x; 𝜇, 𝜎^2) = (1/(𝜎sqrt(2pi)))e^(-(x - 𝜇)^2/(2𝜎^2))"),
        ("Why is learning the weights of a linear regression model an inference problem.", "We can consider the problem to be us trying to infer the most likely weights given the Y's and X's. That is find the best W for P(W|X,Y)"),
        ("What is the main advantage of using the inference to calculate the weights instead of pseudo inverse method.", "Using inference we get error bars on our predictions, 𝜎^2_pred = 𝜎^2_noise + x^T Σ̂_map x"),
        ("How do we solve the inverse problem for linear regression using the inference problem. That is if we know the wieghts W and the outputs Y, how do we find the X's?", "Since Y=XW is \"symmetric\" finding X uses the same process as finding W's"),
        ("Briefly describe the EM process in the context of infering the X's and W's given the Y's",
"""
The EM process is an iterative process:
0. Start by making an intiall guess of all the weights, W', then repeat steps 1 and 2
1. (E step) Treat W' as the true value of the weights and then use them to calculate the most likely set of inputs X'
2. (M step) Treat X' as the true inputs and then use them to calculate the most likely set of weights W'

-  This allows us to do a kind of dimensionality reduction:
   -  From high-dim data (Y), finds the best linear, low-dim embedding (X):
- Also handles missing values correctly :)
"""),


    ]

}
