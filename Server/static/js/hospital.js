var form 
var url
function Assign(){
    form = document.getElementById( "myForm" )
    url = form.getAttribute("action")
    console.log(url)
    // form.addEventListener( "submit", function ( event ) {
    //     event.preventDefault();
    //     showProgress()
    //     sendData();
    // } );
}

function sendData(){
    var XHR = new XMLHttpRequest();
    const FD = new FormData( form );
    XHR.open("POST", url);
    XHR.onreadystatechange = function() {
        if (this.readyState == 4 && this.status == 200) {
            var result = JSON.parse(XHR.responseText)
            console.log(result['Upload Status'])
            document.getElementById("progressDisplay").style.display = "none";
            var res = document.getElementById("res");
            if(result['Upload Status'].slice(0,5) == "Model"){
                res.setAttribute("class","alert alert-success")
            }else{
                res.setAttribute("class","alert alert-danger")
            }
            document.getElementById("resdis").innerHTML = result['Upload Status']
            res.style.display = "block"
        }
    };
    XHR.send(FD);
}


function encrypt(plaintext){
    var key = 21
    var n = 187
    var cipher =[]
    for(let i =0;i<plaintext.length;i++){
     var t = bigInt(plaintext[i].charCodeAt(0)).pow(key).mod(n).valueOf()
    //  console.log(t)
     cipher.push(t)
    }
    // console.log(cipher)
    return cipher.toString()
}

function showProgress(){
    var fileEle = document.getElementById("formFileLg").files[0]
    console.log(fileEle)
    var reader = new FileReader();
     reader.onload = function () {
        var allText = reader.result
        var allTextLines = allText.split(/\r\n|\n/);
        var headers = allTextLines[0].split(',');
        var txt = ""
        for (var i=0; i<allTextLines.length; i++) {
            var data = allTextLines[i].split(',');
            if (data.length == headers.length) {
                var line = ""
                for (var j=0; j<headers.length; j++) {
                    line = line + "," +data[j]
                }
                txt = txt + line + '\n'
            }
        }
        var formEle = document.getElementById("myForm")
        console.log(formEle.children)
        var ele = formEle.children[2]
        ele.value = encrypt(allText)
        // ele.value = "Hello"
        // console.log(ele.value)
        formEle.submit()
        // sendData()
    };
    reader.readAsText(fileEle);  
    // document.getElementById("progressDisplay").style.display = "block";
    // setInterval(()=>{
    //     var bar = document.getElementById("innerprgCus").clientWidth
    //     bar += 10
    //     bar %= 480
    //     s = String(bar)+"px"
    //     document.getElementById("innerprgCus").style.width = s
    // },100)
}

// function sendData() {
//       document.getElementById("res").style.display = "none";
//       const XHR = new XMLHttpRequest();
//       const FD = new FormData( form );
//       XHR.onreadystatechange = function() {
//             if (this.readyState == 4 && this.status == 200) {
//               var result = JSON.parse(XHR.responseText)
//               console.log(result['Upload Status'])
//               document.getElementById("progressDisplay").style.display = "none";
//               var res = document.getElementById("res");
//               if(result['Upload Status'].slice(0,5) == "Model"){
//                   res.setAttribute("class","alert alert-success")
//               }else{
//                   res.setAttribute("class","alert alert-danger")
//               }
//               document.getElementById("resdis").innerHTML = result['Upload Status']
//               res.style.display = "block"
//             }
//        };
//       XHR.open("POST", url,true);
//       XHR.send( FD );
// }



// function sendData() {
//       document.getElementById("res").style.display = "none";
//       const XHR = new XMLHttpRequest();
//       const FD = new FormData( form );
//       XHR.onreadystatechange = function() {
//             if (this.readyState == 4 && this.status == 200) {
//               var result = JSON.parse(XHR.responseText)
//               console.log(result['Upload Status'])
//               document.getElementById("progressDisplay").style.display = "none";
//               var res = document.getElementById("res");
//               if(result['Upload Status'].slice(0,5) == "Model"){
//                   res.setAttribute("class","alert alert-success")
//               }else{
//                   res.setAttribute("class","alert alert-danger")
//               }
//               document.getElementById("resdis").innerHTML = result['Upload Status']
//               res.style.display = "block"
//             }
//        };
//       XHR.open( "POST", url,true);
//       XHR.send( FD );
//     }
function contact(){
  alert("Gmail   : jeff7654321@gmail.com\n"
         +"Contact : 9876543210 ")
}
function aboutUs(){
  alert("Version Number  : 1.0\n"
       +"Author          : Jeffrey Nicholas Y ")
}