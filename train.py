import os, argparse, time
import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
from torchvision import transforms
import random
import torch.backends.cudnn as cudnn
import scipy
import scipy.io as scio
from scipy import stats
from scipy.optimize import curve_fit
from Rank2Score import Rank2Score
from utils.MyDataset import MyDataset, PromptCurriculumBatchSampler
from utils.loss import MSE_Learning, Rank_Learning
from scipy.stats import pearsonr, spearmanr, kendalltau
import model.clip as clip


def set_rand_seed(seed=2001):
    print("Random Seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)       
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  


def estimate(pred, target):
    _, _, pred = logistic_5_fitting_no_constraint(pred, target)
    plcc, _ = pearsonr(pred, target)
    srocc, _ = spearmanr(pred, target)
    krocc, _ = kendalltau(pred, target)
    rmse = np.sqrt(np.mean((pred - target) ** 2))
    results = np.array([plcc, srocc, krocc, rmse])
    return results


def logistic_5_fitting_no_constraint(x, y):
    def func(x, b0, b1, b2, b3, b4):
        logistic_part = 0.5 - np.divide(1.0, 1 + np.exp(b1 * (x - b2)))
        y_hat = b0 * logistic_part + b3 * np.asarray(x) + b4
        return y_hat

    x_axis = np.linspace(np.amin(x), np.amax(x), 100)
    init = np.array([np.max(y), np.min(y), np.mean(x), 0.1, 0.1])
    popt, _ = curve_fit(func, x, y, p0=init, maxfev=int(1e8))
    curve = func(x_axis, *popt)
    fitted = func(x, *popt)

    return x_axis, curve, fitted


def curriculum_krcc(pred, target, threshold):
    """Average KRCC on pairs whose MOS gap satisfies Eq. 15."""
    values = []
    for d in range(target.shape[1]):
        valid = target[:, d] != 0
        gt, pr = target[valid, d], pred[valid, d]
        if gt.size < 2:
            continue
        i, j = np.triu_indices(gt.size, 1)
        keep = np.abs(gt[i] - gt[j]) >= threshold
        if not np.any(keep):
            continue
        concordance = np.sign(gt[i][keep] - gt[j][keep])
        prediction = np.sign(pr[i][keep] - pr[j][keep])
        values.append(np.mean(concordance * prediction))
    return float(np.mean(values)) if values else 0.0

def parse_args():
    """Parse input arguments. """
    parser = argparse.ArgumentParser(description="training")
    parser.add_argument('--gpu', help="GPU device id to use [0]", default=0, type=int)
    parser.add_argument('--dataset_name', type=str, default='MY')
    parser.add_argument('--num_epochs', help='40 for Stage-I and 10 for Stage-II.', default=40, type=int)
    parser.add_argument('--batch_size', help='Batch size.', default=8, type=int)
    parser.add_argument('--test_patch_num', help='Test patch number.', default=1, type=int)
    parser.add_argument('--lr_encoder', default=0.000002, type=float, help='learning rate in the visual encoder')
    parser.add_argument('--lr_others', default=0.0003, type=float, help='learning rate in other parts')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--data_dir', type=str, help='Directory for storing projection images')
    parser.add_argument('--img_length_read', default=6, type=int, help = 'number of the using images')
    parser.add_argument('--n_ctx', default=12, type=int, help = 'number of context vectors')
    parser.add_argument('--output_dir', type=str, default='results',help = 'path to the saved models')
    parser.add_argument('--save_flag', help="Flag of saving trained models", default=True, type=bool)
    parser.add_argument('--loss', type=str,default='rank')
    parser.add_argument('--pretrained_path', type=str)
    parser.add_argument('--class_token_position_level', default='middle', type=str, help = "'middle' or 'end' or 'front'")
    parser.add_argument('--k_fold_num', default=5, type=int, help='default 5-fold for training set')
    parser.add_argument('--results_name', type=str)
    parser.add_argument('--csv_dir', type=str, default='csvfiles/my_info')
    parser.add_argument('--curriculum_epsilon', type=float, default=1e-2)
    parser.add_argument('--rank_margin', type=float, default=0.5)
    parser.add_argument('--contrastive_temperature', type=float, default=2.0)
    parser.add_argument('--contrastive_weight', type=float, default=1.0)
    args = parser.parse_args()
    return args


def extend_args(args):
    args.csc = True  
    args.ctx_init = ""  
    args.prec = "fp32"  
    args.subsample_classes = "all"  

def main(args):
    print('*************************************************************************************************************************')
    cudnn.enabled = True
    save_flag = args.save_flag
    output_dir = args.output_dir
    if save_flag:
        os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    dataset_name = args.dataset_name
    loss_type = args.loss
    
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    img_length_read = args.img_length_read
    test_patch_num = args.test_patch_num

    data_dir = args.data_dir
    results_name = args.results_name

    best_all_object_alignment= np.zeros([args.k_fold_num, 4])
    best_all_attribute_alignment = np.zeros([args.k_fold_num, 4])
    best_all_interaction_alignment = np.zeros([args.k_fold_num, 4])
    best_all_overall_alignment = np.zeros([args.k_fold_num, 4])
    best_all_texture_clarity = np.zeros([args.k_fold_num, 4])
    best_all_texture_aesthetics = np.zeros([args.k_fold_num, 4])
    best_all_geometry_loss= np.zeros([args.k_fold_num, 4])
    best_all_geometry_redundancy = np.zeros([args.k_fold_num, 4])
    best_all_geometry_roughness = np.zeros([args.k_fold_num, 4])
    best_all_overall_visual = np.zeros([args.k_fold_num, 4])
    best_all_3d_authentic = np.zeros([args.k_fold_num, 4])
    best_all_overall_quality = np.zeros([args.k_fold_num, 4])
        
        
    for k_fold_id in range(1,args.k_fold_num + 1):

        print('The current k_fold_id is ' + str(k_fold_id)) 
        
         
        train_filename_list = os.path.join(args.csv_dir, 'train_'+str(k_fold_id)+'.csv')
        test_filename_list = os.path.join(args.csv_dir, 'test_'+str(k_fold_id)+'.csv')
        score_list = [1.0, 2.0, 3.0, 4.0, 5.0]


        quality_dimensions = [
        'object alignment','attribute alignment','interaction alignment' ,'overall alignment', \
        'texture clarity','texture aesthetics','geometry loss','geometry redundancy','geometry roughness','overall visual', \
        '3D authentic','overall quality']

        
        best_object_alignment= np.zeros(4)
        best_attribute_alignment = np.zeros(4)
        best_interaction_alignment = np.zeros(4)
        best_overall_alignment = np.zeros(4)
        best_texture_clarity = np.zeros(4)
        best_texture_aesthetics = np.zeros(4)
        best_geometry_loss= np.zeros(4)
        best_geometry_redundancy = np.zeros(4)
        best_geometry_roughness = np.zeros(4)
        best_overall_visual = np.zeros(4)
        best_3d_authentic = np.zeros(4)
        best_overall_quality = np.zeros(4)
        
        
        dimension_level = ['bad', 'poor', 'fair', 'good', 'excellent']
        
        transformations_train = transforms.Compose([transforms.Resize(224),\
                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])



        transformations_test = transforms.Compose([transforms.Resize(224),\
                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
        
        print('Trainging set: ' + train_filename_list)
        

        model = Rank2Score(device, args, quality_dimensions, dimension_level).to(device)
        
        
        if loss_type == "ft":
            criterion = MSE_Learning(device).to(device)
            pretrained_path = args.pretrained_path.format(fold=k_fold_id)
            state_dict = torch.load(pretrained_path, map_location=device)
            model.load_state_dict(state_dict)
            
            # Stage-II freezes CLIP-Text and the quality-level learner (Sec. 5.2.3).
            for encoder in [model.level_prompt_learner, model.text_encoder]:
                for param in encoder.parameters():
                    param.requires_grad = False
                
            
        elif loss_type == "rank":

            criterion = Rank_Learning(
                device,
                margin=args.rank_margin,
                temperature=args.contrastive_temperature,
                contrastive_weight=args.contrastive_weight,
            ).to(device)
            
            for param in model.text_encoder.parameters():
                param.requires_grad = False
        
            
        clip_encoder_params = model.clip_model.parameters()   
        other_params = [
            p for name, p in model.named_parameters() 
            if 'clip_model' not in name and p.requires_grad
        ]  
        optimizer = torch.optim.AdamW([
            {'params': clip_encoder_params, 'lr': args.lr_encoder},
            {'params': other_params, 'lr': args.lr_others}  
        ], weight_decay=1e-4)
            
            
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
        
            
        print("Ready to train network")
        print('*************************************************************************************************************************')
        
        min_training_loss = 10000
        num_curriculum_prompts = 1
        score_threshold = max(score_list) / 2.0
        previous_srcc = None
        previous_threshold_krcc = None
        
        train_dataset = MyDataset(data_dir=data_dir, datainfo_path=train_filename_list, data_name=dataset_name, img_length_read=img_length_read, transform=transformations_train, patch_num=1)
        test_dataset = MyDataset(data_dir=data_dir, datainfo_path=test_filename_list, data_name=dataset_name, img_length_read=img_length_read, transform=transformations_test, patch_num=1)

        columns = ['Epoch', 'Train_Loss', 'Train_SRCC1', 'Train_SRCC2', 'Train_SRCC3', 'Train_SRCC4', 'Train_SRCC5', 'Train_SRCC6', 'Train_SRCC7', 'Train_SRCC8', 'Train_SRCC9', 'Train_SRCC10', 'Train_SRCC11', 'Train_SRCC12',\
            'Test_SRCC1', 'Test_SRCC2', 'Test_SRCC3', 'Test_SRCC4', 'Test_SRCC5', 'Test_SRCC6', 'Test_SRCC7', 'Test_SRCC8', 'Test_SRCC9', 'Test_SRCC10', 'Test_SRCC11', 'Test_SRCC12', 'Training_time(s)']
        results_df = pd.DataFrame(columns=columns)
        
        print('Epoch\tTrain_Loss\tTrain_SRCC1\tTrain_SRCC2\tTrain_SRCC3\tTrain_SRCC4\tTrain_SRCC5\tTrain_SRCC6\tTrain_SRCC7\tTrain_SRCC8\tTrain_SRCC9\tTrain_SRCC10\tTrain_SRCC11\tTrain_SRCC12\tTest_SRCC1\tTest_SRCC2\tTest_SRCC3\tTest_SRCC4\tTest_SRCC5\tTest_SRCC6\tTest_SRCC7\tTest_SRCC8\tTest_SRCC9\tTest_SRCC10\tTest_SRCC11\tTest_SRCC12\tTraining_time(s)')
        

        for epoch in range(num_epochs):
            
            if loss_type == "rank":
                batch_sampler = PromptCurriculumBatchSampler(
                    train_dataset.name_prompt, batch_size,
                    num_prompts=num_curriculum_prompts, epoch=epoch
                )
                train_loader = torch.utils.data.DataLoader(
                    dataset=train_dataset, batch_sampler=batch_sampler, num_workers=8
                )
            else:
                train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_patch_num , shuffle=False, num_workers=8, drop_last = True)
            n_train = len(train_loader)
            n_train = len(train_loader)
            n_test = len(test_loader)
            model.train()

            start = time.time()
            batch_losses = []
            x_pre = torch.empty((0, len(quality_dimensions)), device=device)
            x_gt = torch.empty((0, len(quality_dimensions)), device=device)
            for i, (imgs, prompts, mos) in enumerate(train_loader):
                
                imgs = imgs.to(device)
                mos = mos.to(device)

                quality_score, feature_fusion = model(imgs, prompts, dataset_name, score_list)

                consistency_threshold = max(0.0, 0.5 - 0.5 * epoch / max(num_epochs, 1))
                loss = criterion(
                    quality_score, mos, feature_fusion, dataset_name,
                    score_threshold=score_threshold if loss_type == "rank" else 0.0,
                    consistency_threshold=consistency_threshold if loss_type == "rank" else 0.0,
                )
                
                batch_losses.append(loss.item())
                x_pre = torch.cat([x_pre,quality_score], dim=0)
                x_gt = torch.cat([x_gt,mos], dim=0)
                
                optimizer.zero_grad()   
                torch.autograd.backward(loss)
                optimizer.step()

            
        
            x_pre_object_alignment = x_pre[:,0].cpu().detach().numpy()
            x_pre_attribute_alignment = x_pre[:,1].cpu().detach().numpy()
            x_pre_interaction_alignment = x_pre[:,2].cpu().detach().numpy()
            x_pre_overall_alignment = x_pre[:,3].cpu().detach().numpy()
            x_pre_texture_clarity = x_pre[:,4].cpu().detach().numpy()
            x_pre_texture_aesthetics = x_pre[:,5].cpu().detach().numpy()
            x_pre_geometry_loss = x_pre[:,6].cpu().detach().numpy()
            x_pre_geometry_redundancy = x_pre[:,7].cpu().detach().numpy()
            x_pre_geometry_roughness = x_pre[:,8].cpu().detach().numpy()
            x_pre_overall_visual = x_pre[:,9].cpu().detach().numpy()
            x_pre_3d_authentic = x_pre[:,10].cpu().detach().numpy()
            x_pre_overall_quality = x_pre[:,11].cpu().detach().numpy()
            
            x_gt_object_alignment = x_gt[:,0].cpu().detach().numpy()
            x_gt_attribute_alignment = x_gt[:,1].cpu().detach().numpy()
            x_gt_interaction_alignment = x_gt[:,2].cpu().detach().numpy()
            x_gt_overall_alignment = x_gt[:,3].cpu().detach().numpy()
            x_gt_texture_clarity = x_gt[:,4].cpu().detach().numpy()
            x_gt_texture_aesthetics = x_gt[:,5].cpu().detach().numpy()
            x_gt_geometry_loss = x_gt[:,6].cpu().detach().numpy()
            x_gt_geometry_redundancy = x_gt[:,7].cpu().detach().numpy()
            x_gt_geometry_roughness = x_gt[:,8].cpu().detach().numpy()
            x_gt_overall_visual = x_gt[:,9].cpu().detach().numpy()
            x_gt_3d_authentic = x_gt[:,10].cpu().detach().numpy()
            x_gt_overall_quality = x_gt[:,11].cpu().detach().numpy()
            
            
            
            valid_mask_attribute = (x_gt_attribute_alignment != 0)
            x_pre_attribute_alignment = x_pre_attribute_alignment[valid_mask_attribute]
            x_gt_attribute_alignment = x_gt_attribute_alignment[valid_mask_attribute]
            
            valid_mask_interaction= (x_gt_interaction_alignment != 0)
            x_pre_interaction_alignment = x_pre_interaction_alignment[valid_mask_interaction]
            x_gt_interaction_alignment = x_gt_interaction_alignment[valid_mask_interaction]

            
            train_SROCC1, _ = stats.spearmanr(x_pre_object_alignment, x_gt_object_alignment)
            train_SROCC2, _ = stats.spearmanr(x_pre_attribute_alignment, x_gt_attribute_alignment)
            train_SROCC3, _ = stats.spearmanr(x_pre_interaction_alignment, x_gt_interaction_alignment)
            train_SROCC4, _ = stats.spearmanr(x_pre_overall_alignment, x_gt_overall_alignment)
            train_SROCC5, _ = stats.spearmanr(x_pre_texture_clarity, x_gt_texture_clarity)
            train_SROCC6, _ = stats.spearmanr(x_pre_texture_aesthetics, x_gt_texture_aesthetics)
            train_SROCC7, _ = stats.spearmanr(x_pre_geometry_loss, x_gt_geometry_loss)
            train_SROCC8, _ = stats.spearmanr(x_pre_geometry_redundancy, x_gt_geometry_redundancy)
            train_SROCC9, _ = stats.spearmanr(x_pre_geometry_roughness, x_gt_geometry_roughness)
            train_SROCC10, _ = stats.spearmanr(x_pre_overall_visual, x_gt_overall_visual)
            train_SROCC11, _ = stats.spearmanr(x_pre_3d_authentic, x_gt_3d_authentic)
            train_SROCC12, _ = stats.spearmanr(x_pre_overall_quality, x_gt_overall_quality)

            if loss_type == "rank":
                current_srcc = float(np.nanmean([
                    train_SROCC1, train_SROCC2, train_SROCC3, train_SROCC4,
                    train_SROCC5, train_SROCC6, train_SROCC7, train_SROCC8,
                    train_SROCC9, train_SROCC10, train_SROCC11, train_SROCC12,
                ]))
                current_krcc = curriculum_krcc(
                    x_pre.cpu().detach().numpy(), x_gt.cpu().detach().numpy(), score_threshold
                )
                if previous_srcc is not None and abs(current_srcc - previous_srcc) < args.curriculum_epsilon:
                    num_curriculum_prompts = min(num_curriculum_prompts + 1, batch_size)
                if previous_threshold_krcc is not None and abs(current_krcc - previous_threshold_krcc) < args.curriculum_epsilon:
                    score_threshold = max(score_threshold - 1.0, 0.0)
                previous_srcc = current_srcc
                previous_threshold_krcc = current_krcc

                
            avg_loss = sum(batch_losses) / n_train
            scheduler.step()

            end = time.time()
            train_time = end - start 

            model.eval()
            y_pre = torch.empty((0, len(quality_dimensions)), device=device)
            y_gt = torch.empty((0, len(quality_dimensions)), device=device)

            with torch.no_grad():
                for i, (imgs, prompts, mos) in enumerate(test_loader):
                    imgs = imgs.to(device)
                    mos = mos.to(device)

                    quality_score, _ = model(imgs, prompts, dataset_name, score_list)
            
                    y_pre = torch.cat([y_pre, quality_score], dim=0)
                    y_gt = torch.cat([y_gt, mos], dim=0)


            
                y_pre_object_alignment = y_pre[:,0].cpu().detach().numpy()
                y_pre_attribute_alignment = y_pre[:,1].cpu().detach().numpy()
                y_pre_interaction_alignment = y_pre[:,2].cpu().detach().numpy()
                y_pre_overall_alignment = y_pre[:,3].cpu().detach().numpy()
                y_pre_texture_clarity = y_pre[:,4].cpu().detach().numpy()
                y_pre_texture_aesthetics = y_pre[:,5].cpu().detach().numpy()
                y_pre_geometry_loss = y_pre[:,6].cpu().detach().numpy()
                y_pre_geometry_redundancy = y_pre[:,7].cpu().detach().numpy()
                y_pre_geometry_roughness = y_pre[:,8].cpu().detach().numpy()
                y_pre_overall_visual = y_pre[:,9].cpu().detach().numpy()
                y_pre_3d_authentic = y_pre[:,10].cpu().detach().numpy()
                y_pre_overall_quality = y_pre[:,11].cpu().detach().numpy()
                
                y_gt_object_alignment = y_gt[:,0].cpu().detach().numpy()
                y_gt_attribute_alignment = y_gt[:,1].cpu().detach().numpy()
                y_gt_interaction_alignment = y_gt[:,2].cpu().detach().numpy()
                y_gt_overall_alignment = y_gt[:,3].cpu().detach().numpy()
                y_gt_texture_clarity = y_gt[:,4].cpu().detach().numpy()
                y_gt_texture_aesthetics = y_gt[:,5].cpu().detach().numpy()
                y_gt_geometry_loss = y_gt[:,6].cpu().detach().numpy()
                y_gt_geometry_redundancy = y_gt[:,7].cpu().detach().numpy()
                y_gt_geometry_roughness = y_gt[:,8].cpu().detach().numpy()
                y_gt_overall_visual = y_gt[:,9].cpu().detach().numpy()
                y_gt_3d_authentic = y_gt[:,10].cpu().detach().numpy()
                y_gt_overall_quality = y_gt[:,11].cpu().detach().numpy()
                
                
                valid_mask_attribute = (y_gt_attribute_alignment != 0)
                y_pre_attribute_alignment = y_pre_attribute_alignment[valid_mask_attribute]
                y_gt_attribute_alignment = y_gt_attribute_alignment[valid_mask_attribute]
                
                valid_mask_interaction= (y_gt_interaction_alignment != 0)
                y_pre_interaction_alignment = y_pre_interaction_alignment[valid_mask_interaction]
                y_gt_interaction_alignment = y_gt_interaction_alignment[valid_mask_interaction]
            
                test_SROCC1, _ = stats.spearmanr(y_pre_object_alignment, y_gt_object_alignment)
                test_SROCC2, _ = stats.spearmanr(y_pre_attribute_alignment, y_gt_attribute_alignment)
                test_SROCC3, _ = stats.spearmanr(y_pre_interaction_alignment, y_gt_interaction_alignment)
                test_SROCC4, _ = stats.spearmanr(y_pre_overall_alignment, y_gt_overall_alignment)
                test_SROCC5, _ = stats.spearmanr(y_pre_texture_clarity, y_gt_texture_clarity)
                test_SROCC6, _ = stats.spearmanr(y_pre_texture_aesthetics, y_gt_texture_aesthetics)
                test_SROCC7, _ = stats.spearmanr(y_pre_geometry_loss, y_gt_geometry_loss)
                test_SROCC8, _ = stats.spearmanr(y_pre_geometry_redundancy, y_gt_geometry_redundancy)
                test_SROCC9, _ = stats.spearmanr(y_pre_geometry_roughness, y_gt_geometry_roughness)
                test_SROCC10, _ = stats.spearmanr(y_pre_overall_visual, y_gt_overall_visual)
                test_SROCC11, _ = stats.spearmanr(y_pre_3d_authentic, y_gt_3d_authentic)
                test_SROCC12, _ = stats.spearmanr(y_pre_overall_quality, y_gt_overall_quality)


                results_object_alignment = estimate(y_pre_object_alignment, y_gt_object_alignment)
                results_attribute_alignment = estimate(y_pre_attribute_alignment, y_gt_attribute_alignment)
                results_interaction_alignment = estimate(y_pre_interaction_alignment, y_gt_interaction_alignment)
                results_overall_alignment = estimate(y_pre_overall_alignment, y_gt_overall_alignment)
                results_texture_clarity = estimate(y_pre_texture_clarity, y_gt_texture_clarity)
                results_texture_aesthetics = estimate(y_pre_texture_aesthetics, y_gt_texture_aesthetics)
                results_geometry_loss = estimate(y_pre_geometry_loss, y_gt_geometry_loss)
                results_geometry_redundancy = estimate(y_pre_geometry_redundancy, y_gt_geometry_redundancy)
                results_geometry_roughness = estimate(y_pre_geometry_roughness, y_gt_geometry_roughness)
                results_overall_visual = estimate(y_pre_overall_visual, y_gt_overall_visual)
                results_3d_authentic = estimate(y_pre_3d_authentic, y_gt_3d_authentic)
                results_overall_quality = estimate(y_pre_overall_quality, y_gt_overall_quality)
                
               

                print('%-3d\t%-8.3f\t%-8.4f\t%-8.4f\t%-8.4f\t%-8.4f\t%-8.4f\t%-8.4f\t%-8.4f\t%-8.4f\t%-8.4f\t%-8.4f\t%-8.4f\t%-8.4f\t%-8.4f\t%-8.4f\t%-8.4f\t%-8.4f\t%-8.4f\t%-8.4f\t%-8.4f\t%-8.4f\t%-8.4f\t%-8.4f\t%-8.4f\t%-8.4f\t%-8.4f' %
                    (epoch + 1, avg_loss, train_SROCC1, train_SROCC2, train_SROCC3, train_SROCC4, train_SROCC5, train_SROCC6, train_SROCC7, train_SROCC8, train_SROCC9, train_SROCC10, train_SROCC11, train_SROCC12, test_SROCC1, test_SROCC2, test_SROCC3, test_SROCC4, test_SROCC5, test_SROCC6, test_SROCC7, test_SROCC8, test_SROCC9, test_SROCC10, test_SROCC11, test_SROCC12, train_time))
                
                new_row = pd.DataFrame([{
                'Epoch': epoch,
                'Train_Loss': avg_loss,
                'Train_SRCC1': train_SROCC1,
                'Train_SRCC2': train_SROCC2,
                'Train_SRCC3': train_SROCC3,
                'Train_SRCC4': train_SROCC4,
                'Train_SRCC5': train_SROCC5,
                'Train_SRCC6': train_SROCC6,
                'Train_SRCC7': train_SROCC7,
                'Train_SRCC8': train_SROCC8,
                'Train_SRCC9': train_SROCC9,
                'Train_SRCC10': train_SROCC10,
                'Train_SRCC11': train_SROCC11,
                'Train_SRCC12': train_SROCC12,
                'Test_SRCC1': test_SROCC1,
                'Test_SRCC2': test_SROCC2,
                'Test_SRCC3': test_SROCC3,
                'Test_SRCC4': test_SROCC4,
                'Test_SRCC5': test_SROCC5,
                'Test_SRCC6': test_SROCC6,
                'Test_SRCC7': test_SROCC7,
                'Test_SRCC8': test_SROCC8,
                'Test_SRCC9': test_SROCC9,
                'Test_SRCC10': test_SROCC10,
                'Test_SRCC11': test_SROCC11,
                'Test_SRCC12': test_SROCC12,
                'Training_time(s)': train_time
                }])
                results_df = pd.concat([results_df, new_row], ignore_index=True)

                if avg_loss < min_training_loss:
                    if save_flag:
                        output_model_name = os.path.join(output_dir, results_name + loss_type + dataset_name + str(k_fold_id) + '.pth')
                        torch.save(model.state_dict(), output_model_name)
                        output_mat_name = os.path.join(output_dir, results_name + loss_type + dataset_name + str(k_fold_id) + '.mat')
                        scio.savemat(output_mat_name,
                                     {'y_pre':y_pre.cpu().detach().numpy(),'y_gt':y_gt.cpu().detach().numpy(),
                                     'x_pre': x_pre.cpu().detach().numpy(),'x_gt': x_gt.cpu().detach().numpy()})


                    best_object_alignment = results_object_alignment
                    best_attribute_alignment = results_attribute_alignment
                    best_interaction_alignment = results_interaction_alignment
                    best_overall_alignment = results_overall_alignment
                    best_texture_clarity = results_texture_clarity
                    best_texture_aesthetics = results_texture_aesthetics
                    best_geometry_loss = results_geometry_loss
                    best_geometry_redundancy = results_geometry_redundancy
                    best_geometry_roughness = results_geometry_roughness
                    best_overall_visual = results_overall_visual
                    best_3d_authentic = results_3d_authentic
                    best_overall_quality = results_overall_quality


                    min_training_loss = avg_loss


        if save_flag:
            output_excel_name =  os.path.join(output_dir, results_name + loss_type + dataset_name + str(k_fold_id) +'.xlsx')
            results_df.to_excel(output_excel_name, index=False)
            print(f"Training results saved to {output_excel_name}")
        
        best_all_object_alignment[k_fold_id-1, :] = best_object_alignment
        best_all_attribute_alignment[k_fold_id-1, :] = best_attribute_alignment
        best_all_interaction_alignment[k_fold_id-1, :] = best_interaction_alignment
        best_all_overall_alignment[k_fold_id-1, :] = best_overall_alignment
        best_all_texture_clarity[k_fold_id-1, :] = best_texture_clarity
        best_all_texture_aesthetics[k_fold_id-1, :] = best_texture_aesthetics
        best_all_geometry_loss[k_fold_id-1, :] = best_geometry_loss
        best_all_geometry_redundancy[k_fold_id-1, :] = best_geometry_redundancy
        best_all_geometry_roughness[k_fold_id-1, :] = best_geometry_roughness
        best_all_overall_visual[k_fold_id-1, :] = best_overall_visual
        best_all_3d_authentic[k_fold_id-1, :] = best_3d_authentic
        best_all_overall_quality[k_fold_id-1, :] = best_overall_quality
       
        print("1: the best val results in the fold {}: PLCC={:.4f}, SROCC={:.4f}, KROCC={:.4f}, RMSE={:.4f}".format(str(k_fold_id), best_object_alignment[0], best_object_alignment[1], best_object_alignment[2], best_object_alignment[3]))       
        print("2: the best val results in the fold {}: PLCC={:.4f}, SROCC={:.4f}, KROCC={:.4f}, RMSE={:.4f}".format(str(k_fold_id), best_attribute_alignment[0], best_attribute_alignment[1], best_attribute_alignment[2], best_attribute_alignment[3]))
        print("3: the best val results in the fold {}: PLCC={:.4f}, SROCC={:.4f}, KROCC={:.4f}, RMSE={:.4f}".format(str(k_fold_id), best_interaction_alignment[0], best_interaction_alignment[1], best_interaction_alignment[2], best_interaction_alignment[3]))
        print("4: the best val results in the fold {}: PLCC={:.4f}, SROCC={:.4f}, KROCC={:.4f}, RMSE={:.4f}".format(str(k_fold_id), best_overall_alignment[0], best_overall_alignment[1], best_overall_alignment[2], best_overall_alignment[3]))
        print("5: the best val results in the fold {}: PLCC={:.4f}, SROCC={:.4f}, KROCC={:.4f}, RMSE={:.4f}".format(str(k_fold_id), best_texture_clarity[0], best_texture_clarity[1], best_texture_clarity[2], best_texture_clarity[3]))       
        print("6: the best val results in the fold {}: PLCC={:.4f}, SROCC={:.4f}, KROCC={:.4f}, RMSE={:.4f}".format(str(k_fold_id), best_texture_aesthetics[0], best_texture_aesthetics[1], best_texture_aesthetics[2], best_texture_aesthetics[3]))
        print("7: the best val results in the fold {}: PLCC={:.4f}, SROCC={:.4f}, KROCC={:.4f}, RMSE={:.4f}".format(str(k_fold_id), best_geometry_loss[0], best_geometry_loss[1], best_geometry_loss[2], best_geometry_loss[3]))
        print("8: the best val results in the fold {}: PLCC={:.4f}, SROCC={:.4f}, KROCC={:.4f}, RMSE={:.4f}".format(str(k_fold_id), best_geometry_redundancy[0], best_geometry_redundancy[1], best_geometry_redundancy[2], best_geometry_redundancy[3]))
        print("9: the best val results in the fold {}: PLCC={:.4f}, SROCC={:.4f}, KROCC={:.4f}, RMSE={:.4f}".format(str(k_fold_id), best_geometry_roughness[0], best_geometry_roughness[1], best_geometry_roughness[2], best_geometry_roughness[3]))       
        print("10: the best val results in the fold {}: PLCC={:.4f}, SROCC={:.4f}, KROCC={:.4f}, RMSE={:.4f}".format(str(k_fold_id), best_overall_visual[0], best_overall_visual[1], best_overall_visual[2], best_overall_visual[3]))
        print("11: the best val results in the fold {}: PLCC={:.4f}, SROCC={:.4f}, KROCC={:.4f}, RMSE={:.4f}".format(str(k_fold_id), best_3d_authentic[0], best_3d_authentic[1], best_3d_authentic[2], best_3d_authentic[3]))
        print("12: the best val results in the fold {}: PLCC={:.4f}, SROCC={:.4f}, KROCC={:.4f}, RMSE={:.4f}".format(str(k_fold_id), best_overall_quality[0], best_overall_quality[1], best_overall_quality[2], best_overall_quality[3]))
        print('*************************************************************************************************************************')
    
    best_mean_object_alignment = np.mean(best_all_object_alignment, axis=0)
    best_mean_attribute_alignment = np.mean(best_all_attribute_alignment, axis=0)
    best_mean_interaction_alignment = np.mean(best_all_interaction_alignment, axis=0)
    best_mean_overall_alignment = np.mean(best_all_overall_alignment, axis=0)
    best_mean_texture_clarity = np.mean(best_all_texture_clarity, axis=0)
    best_mean_texture_aesthetics = np.mean(best_all_texture_aesthetics, axis=0)
    best_mean_geometry_loss = np.mean(best_all_geometry_loss, axis=0)
    best_mean_geometry_redundancy = np.mean(best_all_geometry_redundancy, axis=0)
    best_mean_geometry_roughness = np.mean(best_all_geometry_roughness, axis=0)
    best_mean_overall_visual = np.mean(best_all_overall_visual, axis=0)
    best_mean_3d_authentic = np.mean(best_all_3d_authentic, axis=0)
    best_mean_overall_quality = np.mean(best_all_overall_quality, axis=0)
    print('*************************************************************************************************************************')
    print("1: the mean val results: PLCC={:.4f}, SROCC={:.4f}, KROCC={:.4f}, RMSE={:.4f}".format(best_mean_object_alignment[0], best_mean_object_alignment[1], best_mean_object_alignment[2], best_mean_object_alignment[3]))       
    print("2: the mean val results: PLCC={:.4f}, SROCC={:.4f}, KROCC={:.4f}, RMSE={:.4f}".format(best_mean_attribute_alignment[0], best_mean_attribute_alignment[1], best_mean_attribute_alignment[2], best_mean_attribute_alignment[3]))
    print("3: the mean val results: PLCC={:.4f}, SROCC={:.4f}, KROCC={:.4f}, RMSE={:.4f}".format( best_mean_interaction_alignment[0], best_mean_interaction_alignment[1], best_mean_interaction_alignment[2], best_mean_interaction_alignment[3]))
    print("4: the mean val results: PLCC={:.4f}, SROCC={:.4f}, KROCC={:.4f}, RMSE={:.4f}".format(best_mean_overall_alignment[0], best_mean_overall_alignment[1], best_mean_overall_alignment[2], best_mean_overall_alignment[3]))
    print("5: the mean val results: PLCC={:.4f}, SROCC={:.4f}, KROCC={:.4f}, RMSE={:.4f}".format(best_mean_texture_clarity[0], best_mean_texture_clarity[1], best_mean_texture_clarity[2], best_mean_texture_clarity[3]))       
    print("6: the mean val results: PLCC={:.4f}, SROCC={:.4f}, KROCC={:.4f}, RMSE={:.4f}".format(best_mean_texture_aesthetics[0], best_mean_texture_aesthetics[1], best_mean_texture_aesthetics[2], best_mean_texture_aesthetics[3]))
    print("7: the mean val results: PLCC={:.4f}, SROCC={:.4f}, KROCC={:.4f}, RMSE={:.4f}".format( best_mean_geometry_loss[0], best_mean_geometry_loss[1], best_mean_geometry_loss[2], best_mean_geometry_loss[3]))
    print("8: the mean val results: PLCC={:.4f}, SROCC={:.4f}, KROCC={:.4f}, RMSE={:.4f}".format(best_mean_geometry_redundancy[0], best_mean_geometry_redundancy[1], best_mean_geometry_redundancy[2], best_mean_geometry_redundancy[3]))
    print("9: the mean val results: PLCC={:.4f}, SROCC={:.4f}, KROCC={:.4f}, RMSE={:.4f}".format(best_mean_geometry_roughness[0], best_mean_geometry_roughness[1], best_mean_geometry_roughness[2], best_mean_geometry_roughness[3]))       
    print("10: the mean val results: PLCC={:.4f}, SROCC={:.4f}, KROCC={:.4f}, RMSE={:.4f}".format(best_mean_overall_visual[0], best_mean_overall_visual[1], best_mean_overall_visual[2], best_mean_overall_visual[3]))
    print("11: the mean val results: PLCC={:.4f}, SROCC={:.4f}, KROCC={:.4f}, RMSE={:.4f}".format( best_mean_3d_authentic[0], best_mean_3d_authentic[1], best_mean_3d_authentic[2], best_mean_3d_authentic[3]))
    print("12: the mean val results: PLCC={:.4f}, SROCC={:.4f}, KROCC={:.4f}, RMSE={:.4f}".format(best_mean_overall_quality[0], best_mean_overall_quality[1], best_mean_overall_quality[2], best_mean_overall_quality[3]))
    print('*************************************************************************************************************************')
    
if __name__ == "__main__":
    args = parse_args()
    extend_args(args)
    print(args)
    set_rand_seed()
    if torch.cuda.is_available():
        with torch.cuda.device(args.gpu):
            main(args)
    else:
        main(args)
