%% Convert data to tiff 

for i=79:99

    cmd = ['/Users/benjenkins/Documents/School/ECE 431/AutoSegment/dicom/0000',num2str(i),'.dcm'];
    I = dicomread(cmd);
    image8 = im2uint8(I);
    I2 = imresize(image8,[227 227]);
    %rgbImage = cat(3, I2, zeros(227), zeros(227));
    %rgbImage = ind2rgb(I2, colormap);
    rgbImage=I2(:,:,[1 1 1]);
    size(rgbImage)
    cmd = ['/Users/benjenkins/Documents/School/ECE 431/AutoSegment/CT/tiff/0000',num2str(i),'.tif']
    imwrite(rgbImage, cmd, 'tiff');
end

for i=100:177

    cmd = ['/Users/benjenkins/Documents/School/ECE 431/AutoSegment/dicom/000',num2str(i),'.dcm'];
    I = dicomread(cmd);
    image8 = im2uint8(I);
    I2 = imresize(image8,[227 227]); 
    rgbImage = cat(3, I2, I2, I2);
    cmd = ['/Users/benjenkins/Documents/School/ECE 431/AutoSegment/CT/tiff/000',num2str(i),'.tif']
    imwrite(rgbImage, cmd, 'tiff');
end

%% Resize mask files
for i=156:177

    cmd = ['/Users/benjenkins/Documents/School/ECE 431/AutoSegment/maskPng/',num2str(i),'.png'];
    I = imread(cmd);
    I2 = imresize(I,[227 227]); 
    %In=rgb2gray(I2);
    size(I2)
    cmd = ['/Users/benjenkins/Documents/School/ECE 431/AutoSegment/CT/mask/',num2str(i),'.tif']
    imwrite(I2, cmd, 'tiff');
end



%% Read in all the data in datastores

maskData = imageDatastore('/Users/benjenkins/Documents/School/ECE 431/AutoSegment/mask/', 'IncludeSubfolders',true,'LabelSource','foldernames');
ctData = imageDatastore('/Users/benjenkins/Documents/School/ECE 431/AutoSegment/tiff/', 'IncludeSubfolders',true,'LabelSource','foldernames');


%% Show some of the images in the datastore
for i = 1:20
    subplot(4,5,i);
    imshow(ctData.Files{i});
end


%% Count data?
%99 tiff
%58 masks
ctData.countEachLabel
maskData.countEachLabel


%% Reload everything as one imagestore with masks and tiff files
 
allData = imageDatastore('/Users/benjenkins/Documents/School/ECE 431/AutoSegment/CT/', 'IncludeSubfolders',true,'LabelSource','foldernames');

%% number of data in each
allData.countEachLabel

%% Training files vs testing files

trainingNumFiles = 50;
[trainAllData,testAllData] = splitEachLabel(allData, trainingNumFiles,'randomize');

%% Set the layers

layers = [imageInputLayer([28 28 1])
	 convolution2dLayer(5,20)
	 reluLayer()
	 maxPooling2dLayer(2,'Stride',2)
	 fullyConnectedLayer(10)
	 softmaxLayer()
	 classificationLayer()];

options = trainingOptions('sgdm','MaxEpochs',20, 'InitialLearnRate',0.001);

%% Train the net! Hopefully

rng(2016) % For reproducibility
convnet = trainNetwork(trainAllData,layers,options);

%% Try alexnet?

net = alexnet

%% WOohoo

%% SOmething alexy?

layer = 'fc7';
trainingFeatures = activations(net,trainAllData,layer);
testFeatures = activations(net,testAllData,layer);

%% Get some labels up in here, the two are mask and tiff
trainingLabels = trainAllData.Labels;
testLabels = testAllData.Labels;

%% make a classifier
classifier = fitcecoc(trainingFeatures,trainingLabels);

%% DO IT
predictedLabels = predict(classifier,testFeatures);


%% show??? HOLY CRAP IT WORKED
idx = [1 4 10 18];
figure
for i = 1:numel(idx)
    subplot(2,2,i)

    I = readimage(testAllData,idx(i));
    label = predictedLabels(idx(i));

    imshow(I)
    title(char(label))
    drawnow
end

%% accuracy = 100% AWE YEAH
accuracy = sum(predictedLabels==testLabels)/numel(predictedLabels)


%% Okay let's do this with some actual classifying now
%First step: partner images with correct submasks and some with incorrect
%Lets pair 79 - 115 with correct masks

for i=79:99

    cmd = ['/Users/benjenkins/Documents/School/ECE 431/AutoSegment/dicom/0000',num2str(i),'.dcm'];
    Idcm = dicomread(cmd);
    image8 = im2uint8(Idcm);
    I1 = imresize(image8,[227 227]);
    
    
    cmd = ['/Users/benjenkins/Documents/School/ECE 431/AutoSegment/maskPng/',num2str(i),'.png'];
    Ipng = imread(cmd);
    I2 = imresize(Ipng,[227 227]); 
    %In=rgb2gray(I2);
    size(I2)
    %overLappingImage = cat(3, I2(:, :, 1), I1, I2(:, :, 3));
    overLappingImage = imfuse(I1, I2);
    
    cmd = ['/Users/benjenkins/Documents/School/ECE 431/AutoSegment/CT/Combined/correct/',num2str(i),'.tif']
    imwrite(overLappingImage, cmd, 'tiff');
end

%% 

for i=100:115

    cmd = ['/Users/benjenkins/Documents/School/ECE 431/AutoSegment/dicom/000',num2str(i),'.dcm'];
    Idcm = dicomread(cmd);
    image8 = im2uint8(Idcm);
    I1 = imresize(image8,[227 227]);
    
    
    cmd = ['/Users/benjenkins/Documents/School/ECE 431/AutoSegment/maskPng/',num2str(i),'.png'];
    Ipng = imread(cmd);
    I2 = imresize(Ipng,[227 227]); 
    %In=rgb2gray(I2);
    size(I2)
   %overLappingImage = cat(3, I2(:, :, 1), I1, I2(:, :, 3));
    overLappingImage = imfuse(I1, I2);
    
    cmd = ['/Users/benjenkins/Documents/School/ECE 431/AutoSegment/CT/Combined/correct/',num2str(i),'.tif']
    imwrite(overLappingImage, cmd, 'tiff');
end

for i=156:175

    cmd = ['/Users/benjenkins/Documents/School/ECE 431/AutoSegment/dicom/000',num2str(i),'.dcm'];
    Idcm = dicomread(cmd);
    image8 = im2uint8(Idcm);
    I1 = imresize(image8,[227 227]);
    
    
    cmd = ['/Users/benjenkins/Documents/School/ECE 431/AutoSegment/maskPng/',num2str(i),'.png'];
    Ipng = imread(cmd);
    I2 = imresize(Ipng,[227 227]); 
    %In=rgb2gray(I2);
    size(I2)
    %overLappingImage = cat(3, I2(:, :, 1), I1, I2(:, :, 3));
    overLappingImage = imfuse(I1, I2);
    
    cmd = ['/Users/benjenkins/Documents/School/ECE 431/AutoSegment/CT/Combined/correct/',num2str(i),'.tif']
    imwrite(overLappingImage, cmd, 'tiff');
end

%% Generate false bitmask-image pairs for CT scans 116 - 155
for i = 116:155
    cmd = ['/Users/benjenkins/Documents/School/ECE 431/AutoSegment/dicom/000',num2str(i),'.dcm'];
    Idcm = dicomread(cmd);
    image8 = im2uint8(Idcm);
    I1 = imresize(image8,[227 227]);
    
    
    cmd = ['/Users/benjenkins/Documents/School/ECE 431/AutoSegment/maskPng/',num2str(randi([79 99], 1)),'.png'];
    Ipng = imread(cmd);
    I2 = imresize(Ipng,[227 227]); 
    J = imrotate(I2,randi([1 115], 1),'bilinear','crop');
    
    %overLappingImage = cat(3, J(:, :, 1), I1, J(:, :, 3));
    overLappingImage = imfuse(I1, J);
    cmd = ['/Users/benjenkins/Documents/School/ECE 431/AutoSegment/CT/Combined/incorrect/',num2str(i),'.tif']
    imwrite(overLappingImage, cmd, 'tiff');
end

%% L33t. Now let's train our network on these correct and incorrect image-bitmap pairings
 
allData = imageDatastore('/Users/benjenkins/Documents/School/ECE 431/AutoSegment/CT/Combined', 'IncludeSubfolders',true,'LabelSource','foldernames');

%% number of data in each
allData.countEachLabel

%% Training files vs testing files

trainingNumFiles = 35;
[trainAllData,testAllData] = splitEachLabel(allData, trainingNumFiles,'randomize');

%% Try alexnet?

net = alexnet

%% WOohoo

%% SOmething alexy?

layer = 'fc7';
trainingFeatures = activations(net,trainAllData,layer);
testFeatures = activations(net,testAllData,layer);

%% Get some labels up in here, the two are correct and incorrect
trainingLabels = trainAllData.Labels;
testLabels = testAllData.Labels;

%% make a classifier
classifier = fitcecoc(trainingFeatures,trainingLabels);

%% DO IT
predictedLabels = predict(classifier,testFeatures);


%% show??? YISS
idx = [26 20 10 24];
%idx = [20 21 22 23];
figure
for i = 1:numel(idx)
    subplot(2,2,i)

    I = readimage(testAllData,idx(i));
    label = predictedLabels(idx(i));

    imshow(I)
    title(char(label))
    drawnow
end

%% accuracy = 100% AWE YEAH
accuracy = sum(predictedLabels==testLabels)/numel(predictedLabels)

