var el = x => document.getElementById(x);
 
function showPicker() {
  el("file-input").click();
}
 
function showPicked(input) {
  el("upload-label").innerHTML = input.files[0].name;
  var reader = new FileReader();
  reader.onload = function(e) {
    // el("image-picked").src = e.target.result;
    var canvas = document.getElementById("image-picked");
    canvas.style.backgroundImage = `url(${e.target.result})`;
    canvas.className = "";
    var ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  };
  reader.readAsDataURL(input.files[0]);
}
 
function analyze() {
  var uploadFiles = el("file-input").files;
  if (uploadFiles.length !== 1) alert("Please select a file to analyze!");
 
  el("analyze-button").innerHTML = "Analyzing...";
  var xhr = new XMLHttpRequest();
  var loc = window.location;
  xhr.open("POST", `${loc.protocol}//${loc.hostname}:${loc.port}/analyze`, true);
  xhr.onerror = function() {
    alert(xhr.responseText);
  };
  xhr.onload = function(e) {
    if (this.readyState === 4) {
        // el('image-picked').src = e.target.responseText;
        var canvas = document.getElementById("image-picked");
        var ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
 
        var response = JSON.parse(e.target.responseText);
        var prob = parseFloat(response['prob']);
         
        if (prob > 0.5) {
            var bbox = JSON.parse(response['bbox']);
            var hw = JSON.parse(response['size']);
            var w_mult = canvas.width / hw[1];
            var h_mult = canvas.height / hw[0];
 
            ctx.rect(bbox[0]*w_mult, bbox[1]*h_mult, bbox[2]*w_mult, bbox[3]*h_mult);
            ctx.strokeStyle = "navy";
            ctx.lineWidth = 2;
            ctx.stroke();
 
            el("result-label").innerHTML = `Grave detected. Confidence of detection is ${prob.toFixed(4)}.`;
        }
        else {
            el("result-label").innerHTML = `No grave detected.  Confidence of detection is ${(1-prob).toFixed(4)}.`;
        }
    }
    el('analyze-button').innerHTML = 'Analyze';
  };
 
  var fileData = new FormData();
  fileData.append("file", uploadFiles[0]);
  xhr.send(fileData);
}
