let results = [];
let model;

async function loadModel() {
    model = await tf.loadLayersModel('model/model.json');
    console.log('✅ Model loaded');
}
loadModel();

function addResult(r) {
    results.push(r);
    if (results.length > 5) results.shift();
    if (results.length === 5) predict();
}

function resetResults() {
    results = [];
    document.getElementById('prediction').innerText = 'Chưa có dự đoán';
}

function oneHotEncode(seq) {
    const map = {P: [1,0,0], B: [0,1,0], T: [0,0,1]};
    return seq.flatMap(r => map[r] || [0,0,0]);
}

async function predict() {
    if (!model) return;
    const input = tf.tensor2d([oneHotEncode(results)]);
    const pred = model.predict(input);
    const data = await pred.data();
    const labels = ['P','B','T'];
    let maxIndex = data.indexOf(Math.max(...data));
    document.getElementById('prediction').innerText =
        `Dự đoán: ${labels[maxIndex]} (${(data[maxIndex]*100).toFixed(2)}%)`;
}
