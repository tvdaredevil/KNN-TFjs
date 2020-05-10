import 'colors'
import * as tf from '@tensorflow/tfjs'
import {KNearestNeighbors} from './KNN'
import {Tensor, Rank} from '@tensorflow/tfjs'

// const strToNumberGrid = v => v.split('|').map(l => l.split('').map(Number))
const strToNumberGrid = (v: string) => v.replace(/\|/g, '').split('').map(Number)

/**
 * This method takes the unformatted format of the training set
 * where a 0 represents a White Space and a 1 represents a black space
 * and converts it into a 2D array representation keeping track of its labels
 * @param arr The Input data
 * @returns A 2D Array representation of input data
 */
function unzip<T>(arr: T[][]) {
    const table = Array(arr[0].length).fill(0).map((_, i) => {
        return arr.map(e => e[i])
    })
    return table
}

//= Data declarations =======================================|

/** The training set */
const train_set = [
    [0, '00100|01010|10001|11111|10001'],
    [0, '11100|10010|11100|10010|11100'],
    [0, '01100|10010|10000|10010|01100'],
    [0, '11100|10010|10010|10010|11100'],
    [0, '11100|10000|11100|10000|11100'],
    [1, '11110|10000|11100|10000|10000'],
    [1, '11110|10000|10000|10111|11110'],
    [1, '10001|10001|11111|10001|10001'],
    [1, '11111|00100|00100|00100|11111'],
    [1, '11110|00100|00100|10100|11100'],
].map(([k, v]) => [k, strToNumberGrid(v as string)] as [number, number[]])

/** The test set */
const test_set = [
    '10010|10100|11000|10100|10010',
    '10000|10000|10000|10000|11110',
    '11011|10101|10101|10001|10001',
    '11001|11101|10111|10011|10001',
    '01110|10001|10001|10001|01110',
].map(strToNumberGrid)


/**
 * Finds optimal Neighbor Number
 * @param x - The training dataset
 * @param y - Actual labels
 */
function fit(x: Tensor, y: Tensor):number {
    const maxIter = 10
    const gr = (t: any, col = 'green') => `${t instanceof Tensor?t.dataSync()[0]:t}`[col]
    const red = (t: any, col = 'red') => `${t instanceof Tensor?t.dataSync()[0]:t}`[col]
    let neighbors = 1
    const storeBestNumNeighbor = {}

    for (let i = 0; i < maxIter; i++) {
        const model = new KNearestNeighbors(neighbors,x,y)
        const predictedVals = model.predict(x)
        const trainAcc = tf.metrics.binaryAccuracy(predictedVals, y).dataSync()[0] * 100
        
        if (!(trainAcc in storeBestNumNeighbor)) storeBestNumNeighbor[trainAcc] = neighbors
        neighbors++
        if (!(i % 1)) {
            console.log(`=== Iteration ${red((i+'').padEnd(3))}:  | Train Accuracy: ${gr(trainAcc + '%').underline}`)
            console.log('  Labels Expected:', Array.from(y.dataSync()).join(', ').yellow)
            console.log('  Labels Guessed :', Array.from(tf.round(predictedVals).dataSync()).join(', ').yellow)
            tf.metrics.binaryAccuracy(predictedVals, y).dataSync()[0] * 100
        }
    }
    return storeBestNumNeighbor[Math.max(...Object.keys(storeBestNumNeighbor).map(i => +i))]
}



// Full training class
let [_y, _x] = unzip(train_set)
let [x, y] = [tf.tensor(_x as number[][]), tf.tensor(_y as number[])];

const optimalNumNeighbor = fit(x,y)
console.log('Picking: ', optimalNumNeighbor, ' as the most optimal number of neighbors based on the fit method above')
const model = new KNearestNeighbors(optimalNumNeighbor, x, y)
console.log('Created a model and fit it to the supplied Data and labels\n\n'.yellow)

const pred = model.predict(tf.tensor(test_set)).dataSync()
console.log('Predicted Values are: ', Array.from(pred))

const printImageWithLabel = (img, lable) => {
    const chars = [' ', String.fromCharCode(9608)]
    const pr = arr => '  '+arr.map(c => chars[c]).join('')
    console.log(pr(img.slice(0, 5)).green)
    console.log(pr(img.slice(5, 10)).green)
    console.log(pr(img.slice(10, 15)).green,'     -> ', `${lable}`.cyan)
    console.log(pr(img.slice(15, 20)).green)
    console.log(pr(img.slice(20, 25)).green,'\n')
}
test_set.forEach((img, i) => printImageWithLabel(img, pred[i]))
