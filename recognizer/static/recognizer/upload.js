const radioWhite = document.querySelector("[data-white]")
const radioBlack = document.querySelector("[data-black]")
var fen_text = document.querySelector("[data-fen]")

const fenData = JSON.parse(document.getElementById("fen_data").textContent);

if (radioWhite != null){
    radioWhite.onclick = function(){
        fen_text.innerHTML = fenData.fen_white
    }
}

if (radioBlack != null){
    radioBlack.onclick = function(){
        fen_text.innerHTML = fenData.fen_black
    }
}