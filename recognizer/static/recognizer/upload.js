const radioWhite = document.querySelector("[data-white]")
const radioBlack = document.querySelector("[data-black]")
var fenText = document.querySelector("[data-fen]")
var lichessBtn = document.querySelector("#lichess-btn")

const fenData = JSON.parse(document.getElementById("fen_data").textContent);
const lichessUrls = JSON.parse(document.getElementById("lichess_urls").textContent);

if (radioWhite != null){
    radioWhite.onclick = function(){
        fenText.value = fenData.fen_white
        lichessBtn.href = lichessUrls.play_white_url
    }
}

if (radioBlack != null){
    radioBlack.onclick = function(){
        fenText.value = fenData.fen_black
        lichessBtn.href = lichessUrls.play_black_url
    }
}