const saveModelData = () => {
  const modelConfig = {
    model_to_train: document.getElementById('model_to_train').value,
    masking_image: document.getElementById('masking_image').value,
    edge_detector: document.getElementById('edge_detector').value,
    dataset_path: document.getElementById('dataset-path').value + "/",
  }
  if (modelConfig.dataset_path === ""){
      document.getElementById('errorMsg').innerHTML = "Dataset path is not defined"
        return false
  }
    if (document.getElementById('errorMsg').innerHTML === "The folder contains non-image files.")
        return false
  eel.save_model_config(modelConfig);
};

const saveGANData = () => {
  const modelConfig = {
    lr: document.getElementById('lr').value,
    g_d_lr: document.getElementById('g_d_lr').value,
    adam_op: document.getElementById('adam_op').value,
    b_size: document.getElementById('b_size').value,
    image_size: document.getElementById('image_size').value,
    sigma: document.getElementById('sigma').value,
    gan_l: document.getElementById('gan_l').value,
    max_iterations: document.getElementById('max_iterations').value,
    edge_threshold: document.getElementById('edge_threshold').value,
    l1_loss_w: document.getElementById('l1_loss_w').value,
    fm_loss_w: document.getElementById('fm_loss_w').value,
    content_loss_w: document.getElementById('content_loss_w').value,
    style_loss_w: document.getElementById('style_loss_w').value,
    inpaint_adv_loss_w: document.getElementById('inpaint_adv_loss_w').value,
  }
  eel.save_model_config(modelConfig);
};

const browseResult = () => {
  eel.get_dataset_path();
}

eel.expose(writeFolderPath)

function writeFolderPath(resultPath) {
  const fileselector = document.getElementById('dataset-path');
  fileselector.style.width = resultPath.length * 10 + 'px';
  fileselector.value = resultPath;
}

eel.expose(FolderContentError)

function FolderContentError(error) {
  document.getElementById('errorMsg').innerHTML = error
}

