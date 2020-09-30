"use strict";

// copied from https://stackoverflow.com/questions/9899372
function docReady(fn) {
    // see if DOM is already available
    if (document.readyState === "complete" || document.readyState === "interactive") {
        // call on next available tick
        setTimeout(fn, 1);
    } else {
        document.addEventListener("DOMContentLoaded", fn);
    }
}

docReady(function() {

//  post-process autodoc type hints
var elements = document.querySelectorAll(".field-list p");

const regex = /([a-zA_z_0-9]*) \(([a-zA-Z_\[\]\s,\.0-9]+)\)(.*)/;

Array.prototype.forEach.call(elements, function(el, i){
  // convert to text
  let text = el.textContent;
  // re-apply formatting
  if (text.match(regex)){
    text = text.replace(regex, "<strong>$1</strong> (<em>$2</em>)$3");
    el.innerHTML = text;
  }
});
});
