import fs from 'fs';
import path from 'path';
import * as tf from '@tensorflow/tfjs';

const TOLERANCE = 1e-4; // 0.0001 max absolute difference per class

const localIO = (modelPath) => {
    return {
        load: async () => {
            const modelJson = JSON.parse(fs.readFileSync(modelPath, 'utf8'));
            // tfjs_converter uses weightData array buffer
            const weightsName = modelJson.weightsManifest[0].paths[0];
            const weightsPath = path.join(path.dirname(modelPath), weightsName);
            const weightsData = fs.readFileSync(weightsPath).buffer;
            
            return {
                modelTopology: modelJson.modelTopology,
                weightSpecs: modelJson.weightsManifest[0].weights,
                weightData: weightsData
            };
        }
    };
};

async function runParityTest() {
    const args = process.argv.slice(2);
    if (args.length < 2) {
        console.error("Usage: node parity_test.js <path_to_model_json> <path_to_fixtures_json>");
        process.exit(1);
    }
    
    const modelPath = args[0];
    const fixturesPath = args[1];
    
    console.log(`Loading TFJS model from local file ${modelPath}...`);
    // tfjs_converter produces Layers models
    const model = await tf.loadLayersModel(localIO(modelPath));
    
    console.log(`Loading fixtures from ${fixturesPath}...`);
    const fixtures = JSON.parse(fs.readFileSync(fixturesPath, 'utf8'));
    
    console.log(`\nRunning Parity Test against ${fixtures.length} sequences...`);
    
    let allPassed = true;
    let maxDiffObserved = 0;
    
    for (const fixture of fixtures) {
        const inputTensor = tf.tensor(fixture.input);
        
        // Output from tf.js model
        const outputTensor = model.predict(inputTensor);
        const tfjsProbs = await outputTensor.array();
        
        // Expected from Python
        const pyProbs = fixture.expected_probs;
        
        // Compare
        const pyProbsFlat = pyProbs[0];
        const tfjsProbsFlat = tfjsProbs[0];
        
        const pyClass = pyProbsFlat.indexOf(Math.max(...pyProbsFlat));
        const tfjsClass = tfjsProbsFlat.indexOf(Math.max(...tfjsProbsFlat));
        
        if (pyClass !== tfjsClass) {
            console.error(`\n[FAIL] Class Mismatch on ${fixture.id}!`);
            console.error(`  Python Predicted Class: ${pyClass}`);
            console.error(`  TF.js Predicted Class:  ${tfjsClass}`);
            allPassed = false;
        }
        
        for (let i = 0; i < pyProbsFlat.length; i++) {
            const diff = Math.abs(pyProbsFlat[i] - tfjsProbsFlat[i]);
            if (diff > maxDiffObserved) {
                maxDiffObserved = diff;
            }
            if (diff > TOLERANCE) {
                console.error(`\n[FAIL] Probability mismatch on ${fixture.id}, class ${i}`);
                console.error(`  Python: ${pyProbsFlat[i]}`);
                console.error(`  TF.js:  ${tfjsProbsFlat[i]}`);
                console.error(`  Diff:   ${diff} > ${TOLERANCE}`);
                allPassed = false;
            }
        }
        
        // Cleanup tensors
        inputTensor.dispose();
        outputTensor.dispose();
    }
    
    console.log(`\n==================================================`);
    console.log(`               PARITY TEST RESULTS`);
    console.log(`==================================================`);
    console.log(`Tolerance: ${TOLERANCE}`);
    console.log(`Max Diff Observed: ${maxDiffObserved}`);
    
    if (allPassed) {
        console.log(`\n[PASS] TF.js outputs perfectly match Python outputs!`);
        process.exit(0);
    } else {
        console.error(`\n[FAIL] Parity test failed. The exported model is mathematically divergent.`);
        process.exit(1);
    }
}

runParityTest().catch(err => {
    console.error("Unhandled Error:", err);
    process.exit(1);
});
