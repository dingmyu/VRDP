close all; clear; clc;

seqIds = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7];
frameIds = [40, 80, 120, 40, 80, 120, 40, 80, 120, 40, 80, 120, 40, 80, 120, 40, 80, 120, 40, 80, 120, 40, 80, 120];
numFrames = 40;

maxNumObj = 10;
thres = 0.7;
stepsize = 2;

for i = 1 : length(seqIds)
    i
    seqId = seqIds(i);
    frameId = frameIds(i);
    
    %% Read images
    imgs = {};
    steps = min(numFrames - 1, frameId);
    for j = 0 : steps
        img = imread(['./sim_' sprintf('%05d', seqId) ...
            '/frames/frame_' sprintf('%05d', frameId - j) '.png']);
        
        alpha = 1 - j * (1 / steps);
        img = img * alpha + uint8(ones(320, 480, 3) * 135) * (1 - alpha);
        imgs = [imgs; img];
    end
    
    %% Read masks
    masks = {};
    steps = min(numFrames - 1, frameId);
    for j = 0 : steps
        imgMask = uint8(zeros(320, 480));
        for k = 0 : (maxNumObj - 1)
            fileName = ['./sim_' sprintf('%05d', seqId) ...
                        '/mask/' sprintf('%05d', frameId - j) ...
                        sprintf('/%01d', k) '0001.png'];
            %fileName
            if ~exist(fileName, 'file')
                continue;
            end
            instMask = imread(fileName);
            instMask = instMask(:,:,1);
            %size(imgMask), size(instMask)
            imgMask = max(imgMask, instMask);
        end
        masks = [masks; imgMask];
    end
    
    %% Multiple exposure
    %stepSize = length(imgs) - 1;
    %outDir = ['../2fr/' dataset '/'];
    
    outFile = ['./' sprintf('video%05d_frame%05d_composite.png', seqId, frameId)];
    stroboscopic(imgs(end:-stepsize:1), masks(end:-stepsize:1), outFile, thres);
end
