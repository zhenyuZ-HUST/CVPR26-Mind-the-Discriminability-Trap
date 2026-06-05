import torch
import torch.nn.functional as F
from torch import nn
from utils import *

from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, lora_state_dict, save_lora, load_lora
from loralib import layers as lora_layers

class Metric_Cosine(nn.Module):
    def __init__(self, temperature=10):
        super(Metric_Cosine, self).__init__()
        self.temp = nn.Parameter(torch.tensor(float(temperature)),requires_grad=True)

    def forward(self, logits):
        
        return logits * self.temp   

def evaluate_lora(args, clip_model, loader, dataset):
    clip_model.eval()
    with torch.no_grad():
        template = dataset.template[0] 
        texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
        text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)

    acc = 0.
    tot_samples = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            images, target = images.cuda(), target.cuda()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                image_features = clip_model.encode_image(images)
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)
            cosine_similarity = image_features @ text_features.t()
            acc += cls_acc(cosine_similarity, target) * len(cosine_similarity)
            tot_samples += len(cosine_similarity)
    acc /= tot_samples

    return acc


def run_lora_base(args, clip_model_zs, logit_scale, test_loader):
    if args.dataset == "CropDisease":
        label_names=["Apple___Apple_scab",
                    "Apple___Black_rot",
                    "Apple___Cedar_apple_rust",
                    "Apple___healthy",
                    "Blueberry___healthy",
                    "Cherry_(including_sour)___Powdery_mildew",
                    "Cherry_(including_sour)___healthy",
                    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
                    "Corn_(maize)___Common_rust_",
                    "Corn_(maize)___Northern_Leaf_Blight",
                    "Corn_(maize)___healthy",
                    "Grape___Black_rot",
                    "Grape___Esca_(Black_Measles)",
                    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
                    "Grape___healthy",
                    "Orange___Haunglongbing_(Citrus_greening)",
                    "Peach___Bacterial_spot",
                    "Peach___healthy",
                    "Pepper,_bell___Bacterial_spot",
                    "Pepper,_bell___healthy",
                    "Potato___Early_blight",
                    "Potato___Late_blight",
                    "Potato___healthy",
                    "Raspberry___healthy",
                    "Soybean___healthy",
                    "Squash___Powdery_mildew",
                    "Strawberry___Leaf_scorch",
                    "Strawberry___healthy",
                    "Tomato___Bacterial_spot",
                    "Tomato___Early_blight",
                    "Tomato___Late_blight",
                    "Tomato___Leaf_Mold",
                    "Tomato___Septoria_leaf_spot",
                    "Tomato___Spider_mites Two-spotted_spider_mite",
                    "Tomato___Target_Spot",
                    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
                    "Tomato___Tomato_mosaic_virus",
                    "Tomato___healthy"]
        # label_names=["Apple with Apple Scab",
        #             "Apple with Black Rot",
        #             "Apple with Cedar Apple Rust",
        #             "Apple with Health",
        #             "Blueberry with Health",
        #             "Cherry with Powdery Mildew",
        #             "Cherry with Health",
        #             "Corn with Gray Leaf Spot",
        #             "Corn with Common Rust",
        #             "Corn with Northern Leaf Blight",
        #             "Corn with Health",
        #             "Grape with Black rot",
        #             "Grape with Black Measles",
        #             "Grape with Isariopsis Leaf Spot",
        #             "Grape with Health",
        #             "Orange with Haunglongbing",
        #             "Peach with Bacterial Spot",
        #             "Peach with Health",
        #             "Pepper with Bacterial Spot",
        #             "Pepper with Health",
        #             "Potato with Early Blight",
        #             "Potato with Late Blight",
        #             "Potato with Health",
        #             "Raspberry with Health",
        #             "Soybean with Health",
        #             "Squash with Powdery Mildew",
        #             "Strawberry with Leaf Scorch",
        #             "Strawberry with Health",
        #             "Tomato with Bacterial Spot",
        #             "Tomato with Early Blight",
        #             "Tomato with Late Blight",
        #             "Tomato with Leaf Mold",
        #             "Tomato with Septoria Leaf Spot",
        #             "Tomato with Spider Mites",
        #             "Tomato with Target Spot",
        #             "Tomato with Tomato Yellow Leaf Curl Virus",
        #             "Tomato with Tomato Mosaic Virus",
        #             "Tomato with Health"]
    elif args.dataset == "EuroSAT":
        # label_names=["AnnualCrop",
        #              "Forest",
        #              "HerbaceousVegetation",
        #              "Highway",
        #              "Industrial",
        #              "Pasture",
        #              "PermanentCrop",
        #              "Residential",
        #              "River",
        #              "SeaLake"]
        label_names=["Annual Crop Land",
                    "Forest",
                    "Herbaceous Vegetation Land",
                    "Highway or Road",
                    "Industrial Buildings",
                    "Pasture Land",
                    "Permanent Crop Land",
                    "Residential Buildings",
                    "River",
                    "Sea or Lake",]
    elif args.dataset == "ISIC":
        # label_names=["MEL",
        #              "NV",
        #              "BCC",
        #              "AKIEC",
        #              "BKL",
        #              "DF",
        #              "VASC"]
        label_names=["Melanoma",
                     "Melanocytic Nevus",
                     "Basal Cell Carcinoma",
                     "Actinic Keratosis",
                     "Benign Keratosis",
                     "Dermatofibroma",
                     "Vascular Lesion"]
    #     label_names = [
    #     "Melanoma - Malignant Skin Tumor (Skin Cancer)",
    #     "Melanocytic Nevus - Benign Mole",
    #     "Basal Cell Carcinoma - Non-Melanoma Skin Cancer",
    #     "Actinic Keratosis - Pre-cancerous Sun Damage",
    #     "Benign Keratosis - Non-cancerous Skin Growth",
    #     "Dermatofibroma - Fibrous Skin Lump",
    #     "Vascular Lesion - Abnormal Blood Vessels"
    # ]
    elif args.dataset == "ChestX":
        label_names=["Atelectasis",
                     "Cardiomegaly",
                     "Effusion",
                     "Infiltration",
                     "Mass",
                     "Nodule",
                     "Pneumothorax"]
    import copy
    import numpy as np
    VALIDATION = False
    total_iters = args.epochs * 1
    template = args.template
    zs_acc_list = []
    fine_acc_list = []
    fine40_acc_list = []
    fine80_acc_list = []
    fine120_acc_list = []
    fine160_acc_list = []
    list_lora_layers = apply_lora(args, clip_model_zs)
    clip_model_zs = clip_model_zs.cuda() 
    for idx, (t_all, x_all, y_all) in enumerate(test_loader):
        clip_model = copy.deepcopy(clip_model_zs)
        #scale = nn.Parameter(torch.tensor(10.0))
        metric = Metric_Cosine().cuda()
        #condition_net_parameters = metric.parameters()
        lora_parameters = get_lora_parameters(clip_model)
        parameters_to_update = [{'params': lora_parameters}]
        #parameters_to_update = [{'params': adapter_parameters}]
        optimizer = torch.optim.AdamW(parameters_to_update, weight_decay=1e-2, betas=(0.9, 0.999), lr=2e-4)
        #optimizer = torch.optim.AdamW(get_lora_parameters(clip_model), weight_decay=1e-2, betas=(0.9, 0.999), lr=2e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters * 8, eta_min=1e-6)
        scaler = torch.cuda.amp.GradScaler()

        aug_num = len(x_all)
        test_data = x_all[0].cuda()
        test_label = torch.tensor(np.repeat(range(args.way), args.shot + 15)).cuda()

        all_image = torch.stack(x_all, dim=0).cuda()
        all_labels = torch.tensor(np.repeat(range(args.way), args.shot + 15)).cuda().reshape(args.way, args.shot + 15)
        all_labels = torch.unsqueeze(all_labels, dim=0).repeat(aug_num,1,1) 

        supp_images = all_image[:, :, :args.shot,:,:,:].reshape(args.way * args.shot * aug_num, 3,224,224)

        supp_images_noAug = test_data[:, :args.shot,:,:,:].reshape(args.way * args.shot, 3,224,224)
        query_images = test_data[:, args.shot:,:,:,:].reshape(args.way * 15, 3,224,224)

        supp_label = all_labels[:,:,:args.shot].reshape(-1)

        supp_label_noAug = test_label.reshape(args.way, args.shot + 15)[:,:args.shot].reshape(-1)
        query_label = test_label.reshape(args.way, args.shot + 15)[:,args.shot:].reshape(-1)

        labels_list = y_all[0][:,0]

        class_texts = [label_names[i] for i in labels_list]

        # all_texts = t_all[0][0]

        # a = all_texts
        # class_texts = list(set(a))
        # class_texts.sort(key = a.index)
        

        #################################
        #supp_images = supp_images_noAug
        #supp_label = supp_label_noAug
        ###################################################################
        zs_acc = 0#fsl_test(clip_model, query_images, query_label, class_texts)
        ####################################################################
        with torch.no_grad(): 
            #template = 'a photo of a {}.'#dataset.template[0] 
            full_texts = [template.format(classname.replace('_', ' ')) for classname in class_texts]
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                texts = clip.tokenize(full_texts).cuda()
                class_embeddings = clip_model.encode_text(texts)
            text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)
            
        clip_model.train()
        batch_size = 25
        support_size = supp_images.size(0)
        #total_iters = args.fsl_fine_epoch * args.shots
        count_iters = 0
        while count_iters < total_iters:
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size , batch_size):
                selected_id = torch.from_numpy( rand_id[i: min(i+batch_size, support_size) ]).cuda()
                z_batch = supp_images[selected_id]
                y_batch = supp_label[selected_id] 
                if(args.encoder == 'both'):
                    #template = 'a photo of a {}.'#dataset.template[0] 
                    full_texts = [template.format(classname.replace('_', ' ')) for classname in class_texts]
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        texts = clip.tokenize(full_texts).cuda()
                        class_embeddings = clip_model.encode_text(texts)
                    text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)

                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = clip_model.encode_image(z_batch)
                image_features = image_features/image_features.norm(dim=-1, keepdim=True)

                cosine_similarity = logit_scale * image_features @ text_features.t() 
                loss = F.cross_entropy(cosine_similarity, y_batch)


                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)

                scaler.update()
                scheduler.step()
                count_iters+=1
                save_freq = int(args.epochs / 5)
                if count_iters == save_freq:
                    acc40 = fsl_test(clip_model, query_images, query_label, full_texts)
                    fine40_acc_list.append(acc40)
                    clip_model.train()
                if count_iters == 2 * save_freq:
                    acc80 = fsl_test(clip_model, query_images, query_label, full_texts)
                    fine80_acc_list.append(acc80)
                    clip_model.train()
                if count_iters == 3 * save_freq:
                    acc120 = fsl_test(clip_model, query_images, query_label, full_texts)
                    fine120_acc_list.append(acc120)
                    clip_model.train()
                if count_iters == 4 * save_freq:
                    acc160 = fsl_test(clip_model, query_images, query_label, full_texts)
                    fine160_acc_list.append(acc160)
                    clip_model.train()
                if count_iters == total_iters:
                    break

        ###################################################################
        fine_acc = fsl_test(clip_model, query_images, query_label, full_texts)
        ####################################################################

        zs_acc_list.append(zs_acc)
        fine_acc_list.append(fine_acc)

        if(idx % 1 == 0):
            #print("%d episods: zero shot acc is %g ,finetune acc is %g (ind.)" % (idx, np.mean(np.array(zs_acc_list)), np.mean(np.array(fine_acc_list))))
            #if(args.shot == 1):
            print("%d episods: zero shot acc is %g || 1/5 acc is %g , 2/5 acc is %g , 3/5 acc is %g, 4/5 acc is %g, full acc is %g." % (idx, np.mean(np.array(zs_acc_list)), np.mean(np.array(fine40_acc_list)), np.mean(np.array(fine80_acc_list)), np.mean(np.array(fine120_acc_list)), np.mean(np.array(fine160_acc_list)), np.mean(np.array(fine_acc_list))))
            #else:
            #    print("%d episods: zero shot acc is %g || 200E acc is %g , 400E acc is %g , 600E acc is %g, 800E acc is %g, 1000E acc is %g." % (idx, np.mean(np.array(zs_acc_list)), np.mean(np.array(fine40_acc_list)), np.mean(np.array(fine80_acc_list)), np.mean(np.array(fine120_acc_list)), np.mean(np.array(fine160_acc_list)), np.mean(np.array(fine_acc_list))))

    
def fsl_test(clip_model, query_images, query_label, class_texts):
    clip_model.eval()
    with torch.no_grad(): 
        texts = class_texts
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
        text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)

    acc = 0.
    tot_samples = 0
    with torch.no_grad():
        images, target = query_images.cuda(), query_label.cuda()
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            image_features = clip_model.encode_image(images)
        image_features = image_features/image_features.norm(dim=-1, keepdim=True)
        cosine_similarity = image_features @ text_features.t()
        acc = cls_acc(cosine_similarity, target) * len(cosine_similarity)
        tot_samples = len(cosine_similarity)
        acc /= tot_samples
    return acc

