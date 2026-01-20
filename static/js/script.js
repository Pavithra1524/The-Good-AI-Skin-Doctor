// Example JS for form validations or OTP
function validateOTP() {
    const otpInput = document.getElementById('otp');
    if(otpInput.value.length !== 6) {
        alert('OTP must be 6 digits');
        return false;
    }
    return true;
}

function previewImage(input) {
    const preview = document.getElementById('preview');
    if (input.files && input.files[0]) {
        const reader = new FileReader();
        reader.onload = function(e) {
            preview.src = e.target.result;
        }
        reader.readAsDataURL(input.files[0]);
    }
}
