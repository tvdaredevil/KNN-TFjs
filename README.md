# KNN-TFjs
This is a from scratch implementation of the K-nearest neighbors model in TensorFlowJs

# To Run
- Download this code to a folder
- Open terminal in that folder (ie. `cd` into this folder)
- Setup all dependencies with `npm i`
  - This will install the project dependencies to `node_modules/`
  - Dependencies like: `tensorflow.js`, `nodemon`, `colors`, etc
- To run the code, please run:
  ```shell
  npm start
  ```

# Example
```shell
$ npm start

> knnmodel@1.0.0 start C:\Users\tvdar\Desktop\Final Project\KnnTF
> ts-node ./index


============================
=== Training Iteration 0  :  | Train Accuracy: 100%
  Labels Expected: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
  Labels Guessed : 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
=== Training Iteration 1  :  | Train Accuracy: 80.0000011920929%
  Labels Expected: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
  Labels Guessed : 0, 0, 0, 0, 0, 0, 0, 1, 1, 1
=== Training Iteration 2  :  | Train Accuracy: 80.0000011920929%
  Labels Expected: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
  Labels Guessed : 0, 0, 0, 0, 0, 0, 0, 1, 1, 1
=== Training Iteration 3  :  | Train Accuracy: 69.9999988079071%
  Labels Expected: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
  Labels Guessed : 0, 0, 0, 0, 0, 0, 0, 0, 1, 1
=== Training Iteration 4  :  | Train Accuracy: 69.9999988079071%
  Labels Expected: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
  Labels Guessed : 0, 0, 0, 0, 0, 0, 0, 0, 1, 1
=== Training Iteration 5  :  | Train Accuracy: 69.9999988079071%
  Labels Expected: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
  Labels Guessed : 0, 0, 0, 0, 0, 0, 0, 0, 1, 1
=== Training Iteration 6  :  | Train Accuracy: 69.9999988079071%
  Labels Expected: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
  Labels Guessed : 0, 0, 0, 0, 0, 0, 0, 0, 1, 1
=== Training Iteration 7  :  | Train Accuracy: 50%
  Labels Expected: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
  Labels Guessed : 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
=== Training Iteration 8  :  | Train Accuracy: 50%
  Labels Expected: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
  Labels Guessed : 0, 1, 0, 0, 1, 1, 0, 0, 1, 0
=== Training Iteration 9  :  | Train Accuracy: 50%
  Labels Expected: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
  Labels Guessed : 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
Picking:  1  as the most optimal number of neighbors based on the fit method above
Created a model and fit it to the supplied Data and labels


Predicted Values are:  [ 1, 0, 1, 1, 1 ]
  █  █ 
  █ █  
  ██         ->  1
  █ █  
  █  █ 

  █    
  █    
  █          ->  0
  █    
  ████ 

  ██ ██
  █ █ █
  █ █ █      ->  1
  █   █
  █   █

  ██  █
  ███ █
  █ ███      ->  1
  █  ██
  █   █

   ███ 
  █   █
  █   █      ->  1
  █   █
   ███ 
                                                      
   ```
