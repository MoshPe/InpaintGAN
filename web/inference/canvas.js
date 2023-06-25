const canvas = document.querySelector('#canvas');
const ctx = canvas.getContext('2d');

// only load the image once
const img = new Promise(r => {
  const img = new Image();

  img.src = '../../inference/image.jpeg';
  img.onload = () => r(img);
});

const draw = async () => {
  // resize the canvas to match it's visible size
  canvas.width  = canvas.clientWidth;
  canvas.height = canvas.clientHeight;

  const loaded = await img;
  const iw     = loaded.width;
  const ih     = loaded.height;
  const cw     = canvas.width;
  const ch     = canvas.height;
  const f      = Math.max(cw/iw, ch/ih);

  ctx.setTransform(
    /*     scale x */ f,
    /*      skew x */ 0,
    /*      skew y */ 0,
    /*     scale y */ f,
    /* translate x */ (cw - f * iw) / 2,
    /* translate y */ (ch - f * ih) / 2,
  );

  ctx.drawImage(loaded, 0, 0);
};

window.addEventListener('load', draw);
window.addEventListener('resize', draw);