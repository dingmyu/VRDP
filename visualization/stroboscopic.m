function stroboscopic(imgs, masks, outFile, thres)

sigma = 2;
%denom = zeros(size(imgs{1}));
%numer = zeros(size(imgs{1}));

composite = im2double(imgs{end});
for idx = 1:numel(imgs)
    img = im2double(imgs{idx});
    
    % Quick and dirty way for this particular dataset
    %mask = sum((img-cat(3, 0.86*ones(size(img, 1), size(img, 2)), 0.54*ones(size(img, 1), size(img, 2)), 0.48*ones(size(img, 1), size(img, 2)))).^2, 3)>0.05;
    %mask = cat(3, mask, mask, mask); % for all channels
    mask = im2double(masks{idx});
    mask = imfilter(double(mask), fspecial('gaussian', (sigma * 4 + 1) * [1, 1], sigma));
    mask = mask > thres;
    
    %denom = denom+mask;
    %numer = numer+img.*mask;
    %composite = numer./denom;
    %composite(isnan(composite)) = 1; % white
    %composite(isnan(composite)) = img(isnan(composite));
    for c = 1 : 3
        imgc = img(:, :, c);
        compc = composite(:, :, c);
        compc(mask == 1) = imgc(mask == 1);
        composite(:, :, c) = compc;
    end
end
% Write to image                     
imwrite(composite, outFile);
