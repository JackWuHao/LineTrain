import sys
import math
import numpy
import paddle
import paddle.fluid as fluid
import  argparse

#通过命令给args添加属性
def parse_args():
    parser = argparse.ArgumentParser("fit_a_line")
    parser.add_argument("--enable_ce",action='store_true', help="If set, run the task with continuous evaluation logs." )
    parser.add_argument("--use_gpu", type=bool, default = False, help="Whether to use GPU or not.")
    parser.add_argument("--num_epods", type = int, default = 100, help="number of epochs.")
    args = parser.parse_args()
    return args



def main():
    batch_size = 20
    #如果没有的话会自动下载的，然后再获取数据，所以执行两次
    train_reader = paddle.batch(
        paddle.dataset.uci_housing.train(), batch_size=batch_size)
    test_reader = paddle.batch(
        paddle.dataset.uci_housing.test(), batch_size=batch_size)

    #配置训练程序
    x = fluid.layers.data(name='x', shape=[13],dtype='float32') #定义输入的形状和数据类型(x轴的数据)
    y = fluid.layers.data(name='y', shape = [1], dtype ='float32')
    y_predict = fluid.layers.fc(input = x,size=1,act=None)# 连接输入和输出的全连接层


    main_program =  fluid.default_main_program() #获取默认全局主函数
    start_program = fluid.default_startup_program() # 获取默认全局启动程序
    cost = fluid.layers.square_error_cost(input=y_predict,label=y) #利用标签数据和输出的预测数据估计方差
    avg_loss = fluid.layers.mean(cost) # 对方差求均值，得到平均损失
    test_program  = main_program.clone(for_test=True) #克隆main_program得到test_program
    sgd_optimizer =  fluid.optimizer.SGD(learning_rate=0.001)
    sgd_optimizer.minimize(avg_loss)

    use_cuda = False
    place = fluid.CUDAPlace(0)  if  use_cuda else fluid.CPUPlace()    #指明executor的执行场所
    exe = fluid.Executor(place)

    #定义文件路径保存program
    params_dirname = 'fit_a_line.inference.model'
    num_epods = args.num_epods

    # 训练主循环
    feeder = fluid.DataFeeder(place=place, feed_list=[x,y])
    exe.run(start_program)

    tain_promot = "Train cost"
    test_promot = "test cost"
    step = 0


    exe_text = fluid.Executor(place)

    for  pass_id in range(num_epods):
        for data_train  in  train_reader():
            avg_loss_value, = exe.run(
                main_program,feed=feeder.feed(data_train),fetch_list=[avg_loss]
            )
            # 每10次跑一个记录一次
            if step % 10 == 0 :
                print("%s, Step %d, Cost %f" %
                      (tain_promot, step, avg_loss_value[0]))

           # 每 100次跑一个test
            if step % 100 == 0:
                test_metics = train_test(
                    executor= exe_text,program=test_program,reader=test_reader,feeder=feeder,fetch_list=[avg_loss]
                )
                print("%s, Step %d, Cost %f" %
            (test_promot, step, test_metics[0]))
                 # 如果记录达标停止训练
                if test_metics[0] <10.0:
                    break

            step +=1
            # 数据异常的话，直接退出退出训练
            if math.isnan(float(avg_loss_value[0])):
                sys.exit('got Nan loss, training failed')

        if params_dirname is not None:
            #保存训练数据
            fluid.io.save_inference_model(params_dirname,['x'],[y_predict],exe)


        if args.enable_ce and pass_id == args.num_epods -1 :

            print("kpis\ttrain_cost\t%f" % avg_loss_value[0])
            print("kpis\ttest_cost\t%f" % test_metics[0])



   # 预测(关键) 预测器会从params_dirname 中 读取已经训练好的模型,来对从未遇见的模型进行预测
    infer_exe = fluid.Executor(place)
    inference_scope = fluid.core.Scope()

    with fluid.scope_guard(inference_scope):
          [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(params_dirname,infer_exe) #载入预训练模型
          batch_size = 10

          infer_reader = paddle.batch(paddle.dataset.uci_housing.test(),batch_size=batch_size)

          infer_data = next(infer_reader())

          infer_feat = numpy.array(
              [data[0] for data in infer_data]).astype("float32") # 提取测试集中的数据
          infer_label = numpy.array(
              [data[1] for data in infer_data]).astype("float32") # 提取测试集中的标签
          assert feed_target_names[0] == 'x'   # 断言
          results = infer_exe.run(inference_program,feed={feed_target_names[0]:numpy.array(infer_feat)},fetch_list=fetch_targets)

          print("infer results: (House price)")

          for  idx  ,val  in  enumerate(results[0]):
              print("%d: %0.2f" %(idx, val))
          print("\nground truth:")

          for idx, val in enumerate(infer_label):

              print("%d: %.2f" % (idx, val))

          savc_result(results[0],infer_label)







#创建训练过程
def train_test(executor,program,reader,feeder,fetch_list):

     accumulated = 1 *[0]
     count = 0
     for data_test in reader():
        outs = executor.run(program= program,feed=feeder.feed(data_test),fetch_list=fetch_list)
        accumulated = [x_c[0] + x_c[1][0] for x_c in zip(accumulated, outs)] # 累加测试过程中的损失值
        count += 1  # 累加的测试集中数量
     return [x_d / count  for x_d in accumulated] # 计算平均损值

#保存训练结果(以图表的形式)
def savc_result(points1, points2):
    import matplotlib   # 绘图库
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x1 = [idx for  idx in range(len(points1))]
    y1 = points1
    y2 = points2
    l1 = plt.plot(x1, y1, 'r--',label ='predictions')
    l2 = plt.plot(x1, y2, "g--", label ='GT')
    plt.title('predictions VS GT')
    plt.legend()
    plt.savefig('./image/prediction_gt.png')



if __name__ == '__main__':
    args = parse_args()
    main()