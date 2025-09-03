// This file will handle camera access and can be extended for skeletal overlay
export function startCamera(videoElementId = 'videoElement') {
  const video = document.getElementById(videoElementId);
  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
        video.srcObject = stream;
        video.play();
      })
      .catch(err => {
        console.error('Camera access error:', err);
      });
  }
}
