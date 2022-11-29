<h1 align="center" > Transformer Based Hierarchical Model for Non-Small Cell Lung Cancer Detection & Classification </h1>

<p align='justify'>
The second most prevalent cancer in the world, lung cancer, leads to all other cancers in terms of mortality. The delay in diagnosis, which affects the prognosis and therapy, is one of the leading causes of lung cancer deaths. In order to help pathologists, researchers suggest artificial intelligence-based approaches. We have also attempted to contribute to this noble cause by putting up a unique model for identifying and categorizing non-small cell lung cancer, which is the most prevalent cancer among the other lung cancer types. Our proposed model can detect and classify three types of NSCLC: normal, adenocarcinoma, and squamous cell carcinoma. </p>

<p align="center">
  <img src="https://github.com/ImranNust/LungCancerDetection/blob/main/images/NSCLCTypes.png" />
</p>



Our proposed architecture leaverages the capabilities of convolutional neural network and vision transformers by combining them in an effective manner, as shown below.

<p align="center">
  <img src="https://github.com/ImranNust/LungCancerDetection/blob/main/images/MainModelVer6.png" />
</p>

<hr></hr>
<h2 align="center" > Training the Model </h2>

<p align='justify'>

If you want to use our proposed model for training your own dataset, you can use the [MAIN](https://github.com/ImranNust/LungCancerDetection/blob/main/main.ipynb) file. However, you need to keep a few things in mind while you use our code:

1. We have defined custom layers for patch extraction, patch encoding, metrics, and a few others. All our custom layers are static; that is, they would work for pre-defined batch size. Therefore, you should keep the batch size either 32 or 16. If you want to use other values, you need to modify our code accordingly.
2. We designed our network for images of sizes $256\times256\times3$; however, if you want different sizes, you need to alter our code accordingly and so the patch sizes and other parameters.
3. Our custom metrics functions accept to recieve all three types of classes as inputs; therefore, you need to keep the 'shuffle' parameter true, while generating the train or test iterators for training or testing. 
4. Moreover, for evaluation and prediction, you have to keep the batch size equal to 32. 

</p>

<hr></hr>
<h2 align="center" > Testing the Model </h2>

To test and validate our findings, you can just use the [TestingTheFinalModel](https://github.com/ImranNust/LungCancerDetection/blob/main/TestingTheFinalModel.ipynb) in the Google Colab using the [LINK](https://colab.research.google.com/github/ImranNust/LungCancerDetection/blob/main/TestingTheFinalModel.ipynb). It is advised to use the GPU while running the above code for testing our model.


<hr></hr>
<h2 align="center" > Dataset </h2>

<p align='justify'>

For training, testing, validating, and comparison, we used the [LC25000 LUNG AND COLON HISTOPATHOLOGICAL IMAGE DATASET](https://github.com/tampapath/lung_colon_image_set). The dataset contains histopathological images for lung and colon cancers. Sincer, we are concerned with lung cancer; therefore, here we will talk about it only. There are three subfolders for lung cancer: lung_aca subfolder with 5000 images of lung adenocarcinomas, lung_scc subfolder with 5000 images of lung squamous cell carcinomas, and lung_n subfolder with 5000 images of benign lung tissues.

 </p>

