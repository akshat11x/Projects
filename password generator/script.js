
var output = document.querySelector(".data-length");
// output.innerHTML = slider.value; // Display the default slider value

// // Update the current slider value (each time you drag the slider handle)
// slider.oninput = function() {
//   output.innerHTML = this.value;
// }
const hasLowerCase=document.querySelector("#lowercase")
const hasUpperCase=document.querySelector("#uppercase")
const hasNumbers=document.querySelector("#numbers")
const hasSymbolics=document.querySelector("#symbols")
const copyBtn=document.querySelector("[copy-btn]")
const copyMsg=document.querySelector("[data-copy-msg]")
const passwordDisplay=document.querySelector("[data-password-display]")
let generateBtn = document.querySelector("#generateBtn");
let slider = document.querySelector('input[type=range]');

let passLength=10;
handleSlider();

function handleSlider(){
    slider.value = passLength;
    output.innerText = passLength;
}

slider.addEventListener('input', (event) => {
    passLength = event.target.value;
    handleSlider();
});

function setIndicator(bgColor){
const circle=document.querySelector("#circle");
circle.style.backgroundColor=bgColor;
console.log("I am inside setIndicator")
}


function getRndInt(max,min){
    return Math.floor(Math.random()*(max-min)) + min;
}

function generateRndNum(){
    return getRndInt(0,9);
}

function generateRndLowerCase(){
    return String.fromCharCode(getRndInt(97,123));
}

function generateRndUpperCase(){
    return String.fromCharCode(getRndInt(65,91));
}

function getRndSymbol(){
    const symbols='!@#$%^&*()-+:;/,{}[]'
    const randomIndex = Math.floor(Math.random() * symbols.length);
    return symbols[randomIndex];
}
console.log(getRndSymbol())



function calcStrength(){

   

    let hasUpper=false;
    let hasLower=false;
    let hasNum=false;
    let hasSymbol=false;

    if(hasLowerCase.checked) hasLower=true;
    if(hasUpperCase.checked) hasUpper=true;
    if(hasNumbers.checked) hasNum=true;
    if(hasSymbolics.checked) hasSymbol=true;

    if(hasUpper && hasLower && (hasNum || hasSymbol) && passLength>=8){
        setIndicator("#0f0")
    }else if((hasLower || hasUpper) && (hasNum || hasSymbol) && passLength>=6){
        setIndicator("#ff0");
    }else{
        setIndicator("#f00");
    }


}
async function copyContent(){
    try{
await navigator.clipboard.writeText(passwordDisplay.value);
copyMsg.textContent="copied to clip"

    } catch(e){
copyMsg.textContent="Failed";
    }
    copyMsg.classList.add("active");

    setTimeout(()=>{
        copyMsg.classList.remove("active")
    },2000);
}
copyBtn.addEventListener("click",()=>{
    if (passwordDisplay.value) copyContent();
})

let checkBoxes = document.querySelectorAll("input[type=checkbox]");
console.log(checkBoxes)

checkBoxes.forEach((checkbox)=>{
checkbox.addEventListener('change',handleCheckBoxChange)
})

let checkCount=0;

function handleCheckBoxChange(){
checkCount=0;
checkBoxes.forEach((checkbox)=>{
    if(checkbox.checked) checkCount++;
});
if(passLength<checkCount){
    passLength=checkCount
}
}

let password = "";
generateBtn.addEventListener("click",()=>{
    if(checkCount<=0) return;
    password="";
    if(passLength<checkCount){
        passLength=checkCount;
        handleSlider();
    }
    let funcArr=[];
    if(hasUpperCase.checked){
        funcArr.push(generateRndUpperCase);
    }
    if(hasLowerCase.checked){
        funcArr.push(generateRndLowerCase);
    }
    if(hasNumbers.checked){
        funcArr.push(generateRndNum);
    }
    if(hasSymbolics.checked){
        funcArr.push(getRndSymbol);
    }

    
    for(let i=0;i<funcArr.length;i++){
        password+=funcArr[i]();
    }
    //remaining addition
    for(let i=0;i<passLength-funcArr.length;i++){
        let randIndex=getRndInt(0,funcArr.length)
        password+=funcArr[randIndex]();
    }
//shuffle the password
password=shufflePassword(Array.from(password));

//show in UI
passwordDisplay.value=password;

calcStrength()
})

function shufflePassword(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        const temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
    let str = "";
    array.forEach((el) => (str += el));
    return str;
}