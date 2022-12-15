console.log(symptoms);

// Reference to the buttons
var sub_btn = document.getElementById("sub_btn");
var description = document.getElementById("symptom_description").value;

console.log(description);

if (symptoms[0] === "None") {
    sub_btn.style.backgroundColor = "red";
}
else {
    sub_btn.style.backgroundColor = "green";
}



