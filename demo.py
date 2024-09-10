# 1.def train(conf):
# 2.    logger = util.Logger(conf)
# 3.    if not os.path.exists(conf.checkpoint_dir):  # 用来保存模型
# 4.        os.makedirs(conf.checkpoint_dir)
#
# 5.    model_name = conf.model_name  # FastText
# 6.    dataset_name = "ClassificationDataset"
# 7.    collate_name = "FastTextCollator" if model_name == "FastText" \
#           else "ClassificationCollator"
# 8.    train_data_loader, validate_data_loader, test_data_loader = \
#           get_data_loader(dataset_name, collate_name, conf)  # 数据预处理，获取DataLoader类对象
#       # 是一个ClassificationDataset对象，只执行了__init__函数，加载了{key: index}和{index: key}
#       # 有__getitem__函数，可以用[]调用
#       # {key: index}和{index: key}两种字典不为空，调用__getitem__函数时返回空
# 9.    empty_dataset = globals()[dataset_name](conf, [])
# 10.   model = get_classification_model(model_name, empty_dataset, conf)  # 设置模型
# 11.   loss_fn = globals()["ClassificationLoss"](
#            label_size=len(empty_dataset.label_map), loss_type=conf.train.loss_type)  # 设置损失函数 BCEWITHLOGITS
# 12.   optimizer = get_optimizer(conf, model)  # 设置优化器ADAM
# 13.   evaluator = cEvaluator(conf.eval.dir)  # 设置计算准确率的各项指标
# 14.   trainer = globals()["ClassificationTrainer"](
#            empty_dataset.label_map, logger, evaluator, conf, loss_fn)  # 有准确率和损失函数
#
# 15.   best_epoch = -1
# 16.   best_performance = 0
# 17.   model_file_prefix = conf.checkpoint_dir + "/" + model_name
# 18.   for epoch in range(conf.train.start_epoch,
#                          conf.train.start_epoch + conf.train.num_epochs):  # 迭代训练
# 19.       start_time = time.time()
# 20.       trainer.train(train_data_loader, model, optimizer, "Train", epoch)
# 21.       trainer.eval(train_data_loader, model, optimizer, "Train"