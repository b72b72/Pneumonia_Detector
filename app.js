function sendImage() {
  const input = document.getElementById('xrayInput');
  const file = input.files[0];

  if (!file) {
    alert('Please upload an image.');
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  fetch("/predict", {
    method: "POST",
    body: formData
  })
  .then(response => response.json())
  .then(data => {
    document.getElementById("prediction").textContent = data.prediction;
    document.getElementById("truth").textContent = data.truth;
  })
  .catch(error => {
    console.error("Error:", error);
  });

  const reader = new FileReader();
  reader.onload = function (e) {
    document.getElementById("xrayImage").src = e.target.result;
  };
  reader.readAsDataURL(file);
}
