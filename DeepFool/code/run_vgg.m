% to setup: install matconvnet-1.0-beta23 and download the imagenet-vgg-f
% pretrained model
% clear
% %% load pretrained model
% net = load('imagenet-vgg-f.mat');
% net = vl_simplenn_tidy(net); % add compatibility to newer versions of MatConvNet
% net.layers(end) = []; % the final softmax (loss) layer should be removed in order to prevent numerical instabilities
%%
configure_vgg
% N_all: number of images in the dataset
% imgfile: path to the file containing images
% imgnames: N_all x 1 cell array containing the image names
%%
N = 100; 
see_imgs = true;
if ~exist('r_norm', 'var')
    r_norm = zeros(1, N);
    img_norm = zeros(1, N);
end
for i_img = N
    fprintf('\nProcessing Image %i (%i in total)', i_img, N);
    % image loading, preprocessing to fit the model
    im = imread([imgfile, imgnames{i_img}, '.jpeg']);
    im = single(imresize(im, net.meta.normalization.imageSize(1:2)));
    im_nor = im - net.meta.normalization.averageImage;
    % fool the convnet
    paras.class_k = 0; % set k non-zero to make the program reach a specific class
    paras.norm_p = 2; % choose the p value for p-norm
    paras.overshoot = 0.02; % make sure we always reach a adverserial pertubation
    tic
    [ r, l_fool, l_org ] = deepfool_vgg(im_nor, net, paras);
    toc
    rs(:, i_img) = r(:);
    imgs(:, i_img) = im_nor(:);
    r_norm(i_img) = norm(r(:), 2);
    img_norm(i_img) = norm(im_nor(:), 2);
    % visualization
    if see_imgs
        figure(i_img);
        subplot(1,3,1);
        imagesc(im/255);
        title(['Original label: ', num2str(l_org)]);
        set(gca,'xtick',[],'ytick',[])
        subplot(1,3,2);
        imagesc(r); 
        title('Perturbation [scaled]');
        set(gca,'xtick',[],'ytick',[])
        subplot(1,3,3);
        imagesc((im+r)/255);
        title(['Perturbed label: ', num2str(l_fool)]);
        set(gca,'xtick',[],'ytick',[])
        savefig([imgsave, 'fool_', imgnames{i_img}, '_vis.fig']);
        saveas(gcf, [imgsave, 'fool_', imgnames{i_img}, '_vis.jpeg']);
    end
    % save data
    imwrite((im+r)/256, [imgsave, 'fool_', imgnames{i_img}, '.jpeg']);
    save([imgsave, dataset, '.mat'], 'rs', 'r_norm', 'imgs', 'img_norm');
end