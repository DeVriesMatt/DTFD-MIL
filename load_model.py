


def extract_feats_for_heatmap():
    params = parser.parse_args()

    in_chn = 1024

    classifier = Classifier_1fc(params.mDim, params.num_cls, params.droprate).to(params.device)
    attention = Attention(params.mDim).to(params.device)
    dimReduction = DimReduction(in_chn, params.mDim, numLayer_Res=params.numLayer_Res).to(params.device)
    attCls = Attention_with_Classifier(L=params.mDim, num_cls=params.num_cls, droprate=params.droprate_2).to(
        params.device)

    save_path = params.saved_model_path
    checkpoint = torch.load(save_path, map_location='cuda')

    classifier.load_state_dict(checkpoint['classifier'])

    attention.load_state_dict(checkpoint['attention'])
    dimReduction.load_state_dict(checkpoint['dim_reduction'])
    attCls.load_state_dict(checkpoint['att_classifier'])
    with open(params.mDATA0_dir_train0, 'rb') as f:
        mDATA_train = pickle.load(f)
    with open(params.mDATA0_dir_val0, 'rb') as f:
        mDATA_val = pickle.load(f)
    with open(params.mDATA_dir_test0, 'rb') as f:
        mDATA_test = pickle.load(f)
    #
    # SlideNames_train, FeatList_train, Label_train = reOrganize_mDATA(mDATA_train)
    # SlideNames_val, FeatList_val, Label_val = reOrganize_mDATA(mDATA_val)
    # SlideNames_test, FeatList_test, Label_test = reOrganize_mDATA_test(mDATA_test)

    SlideNames_train, FeatList_train, Label_train = reOrganize_mDATA_v2(mDATA_train)
    SlideNames_val, FeatList_val, Label_val = reOrganize_mDATA_v2(mDATA_val)

    # TODO: Change this whenchanging datasets.
    SlideNames_test, FeatList_test, Label_test = reOrganize_mDATA_test_v2(mDATA_test)

    logits, Y_prob, Y_hat, A, _ = infer_single_slide(
        classifier=classifier, dimReduction=dimReduction,
        attention=attention,
        UClassifier=attCls,
        mDATA_list=([SlideNames_test[0]], [FeatList_test[0]], [Label_test[0]]),
        # TODO: Changed to only see first slide for extractning heatmaps
        criterion=None, params=params, numGroup=params.numGroup_test,
        total_instance=params.total_instance_test,
        distill=params.distill_type)

    return logits, Y_prob, Y_hat, A
