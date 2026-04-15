import argparse
import os
import json
import csv
import multiprocessing
from collections import defaultdict
import time
from datetime import datetime
import re

from cv_paths import CVPaths

PATHS = CVPaths.from_file(__file__)

def iter_tracking_rows_grouped_by_frame(csv_file_path):
    with open(csv_file_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        current_frame = None
        frame_rows = []

        for row in reader:
            try:
                frame_id = int(float(row["frame_id"]))
                row["track_id"] = int(float(row["track_id"]))
                row["center_x"] = float(row["center_x"])
                row["center_y"] = float(row["center_y"])
                row["cls"] = int(float(row["cls"]))
            except (KeyError, TypeError, ValueError):
                continue

            if current_frame is None:
                current_frame = frame_id

            if frame_id != current_frame:
                yield current_frame, frame_rows
                current_frame = frame_id
                frame_rows = []

            frame_rows.append(row)

        if current_frame is not None:
            yield current_frame, frame_rows


#定义一个子进程的工作函数
def CountVechil(CamInfo):
    #将元组数据解包
    point_list,point_list1,point_list2,gate_orientation,CsvFilePath,CountCsvFile=CamInfo
    TempCountCsvFile = f"{CountCsvFile}.part"
    DoneFlagFile = f"{CountCsvFile}.ok"
    count_dir = os.path.dirname(CountCsvFile)
    if count_dir:
        os.makedirs(count_dir, exist_ok=True)
    #创建一个字典，用于存储每个id的历史轨迹
    track_history = defaultdict(lambda: [])
    #创建一个字典，用于存储每个id的触线记录，这个触线历史是临时的，也就是那一帧的触线记录
    touch_history=defaultdict(lambda:[])
    #创建一个字典，用于存储每个id的触线历史
    has_record_history=defaultdict(lambda:[])
    #以写入模式打开这个记录触线信息的csv文件
    if os.path.exists(TempCountCsvFile):
        os.remove(TempCountCsvFile)
    with open(TempCountCsvFile,"w",newline="",buffering=8192) as f:
        #创建一个csv写入对象
        writer=csv.writer(f)
        #写入表头元素
        writer.writerow(["frame_id","track_id","cls","gate_index"])
        #按frame_id分组遍历df
        last_check_time=time.time()
        for frame_id,group_data in iter_tracking_rows_grouped_by_frame(CsvFilePath):
            #每处理五分钟打印一次
            if frame_id>0 and frame_id % 9000 == 0:
                #计算已经处理的分钟数和最近五分钟处理用时
                processed_minutes=round(frame_id/1800,2)
                elapsed_minutes=round((time.time()-last_check_time)/60,2)
                #更新时间检查点
                last_check_time=time.time()
                print(f"csv文件 {os.path.basename(CsvFilePath).split('.')[0]} 已处理 {processed_minutes} 分钟，最近5分钟耗时 {elapsed_minutes} 分钟")
            #遍历当前帧的所有检测结果
            for row in group_data:
                #获取当前检测结果的id
                track_id=row['track_id']
                #获取当前检测结果的中心坐标
                x_center=row['center_x']
                y_center=row['center_y']
                #获取类别信息
                cl_name=row['cls']
                #获取track id对应的轨迹值，是一个列表
                track=track_history[track_id]
                #获取track_id对应的touch_list值，是一个列表
                touch_list=touch_history[track_id]
                has_record=has_record_history[track_id]
                #如果是id是第一次出现，是没有上一帧数据的，我们直接continue
                if len(track) == 0:
                    track.append((x_center,y_center))
                    touch_history[track_id]=touch_list
                    #a list to record this id obeject has touched which line to avoid recur in triple line record
                    has_record=[False for _ in range(len(point_list))]
                    has_record_history[track_id]=has_record
                    continue
                #获取上一帧的中心点
                last_center_xy=track[-1]
                #计算当前点与上一个点的叉积结果
                #position_product_list类似于[[-1.11,2.22],3,-1,None]
                position_product_list=judge_slide((x_center,y_center),point_list)
                position_product_list1=judge_slide((x_center,y_center),point_list1)
                position_product_list2=judge_slide((x_center,y_center),point_list2)
                last_product_list=judge_slide(last_center_xy,point_list)
                last_product_list1=judge_slide(last_center_xy,point_list1)
                last_product_list2=judge_slide(last_center_xy,point_list2)
                #调用触线判别函数，得到一个关于前后两帧是否与各个门触线的列表
                touch_list=touch_line(gate_orientation,point_list,position_product_list,last_product_list,(x_center,y_center),last_center_xy,touch_list)
                touch_list1=touch_line(gate_orientation,point_list1,position_product_list1,last_product_list1,(x_center,y_center),last_center_xy,touch_list)
                touch_list2=touch_line(gate_orientation,point_list2,position_product_list2,last_product_list2,(x_center,y_center),last_center_xy,touch_list)
                #cobine 3 touch list to one
                touch_list=combine_touchList(touch_list,touch_list1,touch_list2,CountCsvFile)
                #根据给定的touch_list，去判断是否写入触线文件
                info_index=0
                for touch_bool in touch_list:
                    #这里要分两种情况，一种情况是touch_bool里是一个列表，对应的是一个门里有两条标注线
                    #另一种情况touch_bool里直接是一个bool值，对应的是一个门里只有一条标注线的情况
                    if isinstance(touch_bool,list):
                        #如果是列表说明门里有两条标注线，我们可以直接在这里做两条线特殊逻辑
                        #就是要标注info，必须满足第一个元素是True，如果第一个元素是False，就算第二个元素是True也不能计入相应info
                        if touch_bool[0] and touch_bool[1] and has_record[info_index]==False:
                            #如果第一个元素是True，第二个元素也是True，那么说明确实过线了，把四个元素写入csv文件
                            writer.writerow([frame_id,track_id,cl_name,info_index+1])
                            #把这个id对应的has_record列表的这个元素设为True
                            has_record[info_index]=True
                    elif isinstance(touch_bool,bool):
                        #如果是bool值说明门里只有一条标注线，我们可以直接在这里做一条线特殊逻辑
                        if touch_bool and has_record[info_index]==False:
                            #如果是True，说明过线了，把这个元素写入csv文件
                            writer.writerow([frame_id,track_id,cl_name,info_index+1])
                            #把这个id对应的has_record列表的这个元素设为True
                            has_record[info_index]=True
                    info_index+=1
                #如果轨迹长度超过了4帧，就删除最早的那帧
                if len(track) > 4:
                    track.pop(0)
                    #也需要删除最早的那帧是否触线列表
                    if touch_history[track_id]:
                        touch_history[track_id].pop(0)
                #将当前帧的检测结果加入到历史轨迹中
                track.append((x_center,y_center))
                #将更新后的是否触线列表更新到对应的touch_history中
                touch_history[track_id]=touch_list

    os.replace(TempCountCsvFile, CountCsvFile)
    with open(DoneFlagFile,"w",encoding="utf-8") as f:
        f.write("ok\n")


def combine_touchList(touch_list,touch_list1,touch_list2,CountCsvFile):
        """
        目标是将三个形如[[True,False],True,[False,False]]，但是中间值各异的值各异的touch_list合成为一个列表
        返回一个统一的touch_list
        """
        index=0
        #for 循环去寻找那些碰到线的元素
        for touch_bool1,touch_bool2 in zip(touch_list1,touch_list2):
            #这里要分两种情况，一种情况是touch_bool里是一个列表，对应的是一个门里有两条标注线
            #另一种情况touch_bool里直接是一个bool值，对应的是一个门里只有一条标注线的情况
            if isinstance(touch_bool1,list):
                #if there is a different means that one is false one is Ture,we make it as true
                if touch_bool1[0] != touch_bool2[0]:
                   touch_list1[index][0]=True
                if touch_bool1[1] != touch_bool2[1]:
                    touch_list1[index][1]=True
                index+=1
                    
            elif isinstance(touch_bool1,bool):
                #如果是bool值说明门里只有一条标注线，我们可以直接在这里做一条线特殊逻辑
                if touch_bool1 != touch_bool2:
                    touch_list1[index]=True
                index+=1
            else:
                print(f"touch_lsit数据格式错误:{CountCsvFile}")
                raise ValueError
        index=0
        #for 循环去寻找那些碰到线的元素
        for touch_bool,touch_bool1 in zip(touch_list,touch_list1):
            #这里要分两种情况，一种情况是touch_bool里是一个列表，对应的是一个门里有两条标注线
            #另一种情况touch_bool里直接是一个bool值，对应的是一个门里只有一条标注线的情况
            if isinstance(touch_bool1,list):
                #if there is a different means that one is false one is Ture,we make it as true
                if touch_bool[0] != touch_bool1[0]:
                   touch_list[index][0]=True
                if touch_bool[1] != touch_bool1[1]:
                    touch_list[index][1]=True
                index+=1
                    
            elif isinstance(touch_bool1,bool):
                #如果是bool值说明门里只有一条标注线，我们可以直接在这里做一条线特殊逻辑
                if touch_bool != touch_bool1:
                    touch_list[index]=True
                index+=1
            else:
                print(f"touch_lsit数据格式错误:{CountCsvFile}")
                raise ValueError
        return touch_list

def touch_line(gate_orientation,gate_point_list,Position_product_list,Last_product_list,xy_center_turple,last_center_xy,last_touch_list):
        """
        这个函数用于判断某一个目标检测框前后两帧是否碰到图像上标注的某一根线了
        输入：gate_orientation是门的方向列表
        gate_point_list是门的标注线点坐标列表
        Position_product_list是当前帧的目标检测框中心点坐标与各个门叉积的列表
        Last_product_list是上一帧的目标检测框中心点坐标与各个门中的标注线叉积的列表
        xy_center_turple是当前帧的目标检测框中心点坐标,last_center_xy是上一帧的目标检测框中心点坐标
        last_touch_list:上一帧得到的touch_list
        输出：一个列表,列表中的每个元素是一个布尔值,且与get_point函数返回的门的标注线一一对应;如果碰到了返回True，否则返回False
        然后输出的列表可能是[[True,False],True,[False,False]]
        """
        #一个列表用于记录前后两帧的连线是否触碰到了任何一条标注线
        touch_list=[]
        #一个指针用于获取当前叉积结果对应的门的标注线点坐标
        gate_point_ptr1=0
        for P_product,L_product in zip(Position_product_list,Last_product_list):
            #获取标注线元组
            gate_point1=gate_point_list[gate_point_ptr1]
            #获取当前门的方向
            gate_orientation1=gate_orientation[gate_point_ptr1]
            if (not gate_point1) or (P_product is None) or (L_product is None):
                # 根据门的方向类型返回对应的 False 结构
                if isinstance(gate_orientation1, list):
                    touch_list.append([False] * len(gate_orientation1)) # 双线门返回 [False, False]
                else:
                    touch_list.append(False) # 单线门返回 False
                gate_point_ptr1 += 1
                continue
            #分两种情况，一种情况是取到了一个包含两元素的列表，一种情况是直接取到了一个值
            if isinstance(P_product,list):
                tem_group=[]
                #另一个指针，同样用于获取当前叉积结果对应的门的标注线的点的坐标
                gate_point_ptr2=0
                #情况1，取到的那个gate是包含两条标注线的
                for P_sub,L_sub in zip(P_product,L_product):
                    #获取标注线元组
                    gate_point2=gate_point1[gate_point_ptr2]
                    #标注线的四个坐标
                    x1,y1,x2,y2=gate_point2
                    #获取当前门的方向,值是1或者-1
                    gate_orientation2=gate_orientation1[gate_point_ptr2] if isinstance(gate_orientation1,list) else gate_orientation1

                    #第一关要判断的是前后两帧中心点是否分别在计数线的两侧(一般情况)
                    if (P_sub<0 and L_sub>0) or (P_sub>0 and L_sub<0):
                        #第二关是要判断计数线的两个端点是否中心点连线的两侧
                        #拿到目标检测框中心点的四个坐标
                        a1,a2=xy_center_turple
                        b1,b2=last_center_xy
                        #现在要构造三个向量，首先要构造中心点连接向量AB
                        AB=(b1-a1,b2-a2)
                        #另外两个向量是计数线的两个端点 与xy_center_turple组成的向量
                        AP_1=(x1-a1,y1-a2)
                        AP_2=(x2-a1,y2-a2)
                        #然后现在分别计算AB与AP1，AB与AP2的叉积
                        cross_product1=AB[0]*AP_1[1]-AB[1]*AP_1[0]
                        cross_product2=AB[0]*AP_2[1]-AB[1]*AP_2[0]
                        #如果两个叉积的符号也是不同的，说明过了第二关，目标检测框的连线确实碰到了计数线，然后就开始判断该线是从什么方向
                        #接触计数线的，正常的触线，有两个判断
                        if cross_product1*cross_product2<0:
                            """
                            如果上一帧中心点叉积为负，当前中心点叉积为正，对应原始的方向应该是-1，就是顺时针方向进入
                            如果上一帧中心点叉积为正，当前中心点叉积为负，对应原始方向应该是1，是逆时针方向进入
                            """
                            if L_sub<0 and P_sub>0:
                                #这时候说明车是顺时针碰到的线，我们要看看原始门方向是不是-1
                                #如果是-1就说明是符合我们定义的方向进入的
                                if gate_orientation2==-1:
                                    tem_group.append(True)
                                else:
                                    #这里要做一个判断，因为是两线的门，所以你需要看看进入线是否为False，False才能append（False）进去
                                    #如果两线门的进入线已经为True了，你就不应该把它判负
                                    if not last_touch_list or (last_touch_list and last_touch_list[gate_point_ptr1][0]==False) or gate_point_ptr2!=0:
                                        tem_group.append(False)
                                    else:
                                        tem_group.append(True)
                                gate_point_ptr2+=1
                            elif L_sub>0 and P_sub<0:
                                #这时候说明车是逆时针碰到的线，我们要看看原始门方向是不是1
                                #如果是1就说明是符合我们定义的方向进入的
                                if gate_orientation2==1:
                                    tem_group.append(True)
                                else:
                                    #这里要做一个判断，因为是两线的门，所以你需要看看进入线是否为False，False才能append（False）进去
                                    #如果两线门的进入线已经为True了，你就不应该把它判负
                                    if not last_touch_list or (last_touch_list and last_touch_list[gate_point_ptr1][0]==False) or gate_point_ptr2!=0:
                                        tem_group.append(False)
                                    else:
                                        tem_group.append(True)
                                gate_point_ptr2+=1
                        else:
                            #这里要做一个判断，因为是两线的门，所以你需要看看进入线是否为False，False才能append（False）进去
                            #如果两线门的进入线已经为True了，你就不应该把它判负
                            if not last_touch_list or (last_touch_list and last_touch_list[gate_point_ptr1][0]==False) or gate_point_ptr2!=0:
                                tem_group.append(False)
                                gate_point_ptr2+=1
                            else:
                                tem_group.append(True)
                                gate_point_ptr2+=1
                    #当前点落在计数线或其延长线的情况
                    elif P_sub==0.0:
                        #如果当前点落在计数线或其延长线的情况，判断当前点是否在计数线的延长线上
                        if on_segment((x1,y1),(x2,y2),xy_center_turple) and L_sub!=0.0:
                            #判断一下当前点的进入方向
                            if L_sub<0:
                                #如果上一帧与线叉积结果为负数，说明顺时针进入的，我们要看看原始的门方向是不是要计入顺时针进入
                                if gate_orientation2==-1:
                                    tem_group.append(True)
                                else:
                                    #这里要做一个判断，因为是两线的门，所以你需要看看进入线是否为False，False才能append（False）进去
                                    #如果两线门的进入线已经为True了，你就不应该把它判负
                                    if not last_touch_list or (last_touch_list and last_touch_list[gate_point_ptr1][0]==False) or gate_point_ptr2!=0:
                                        tem_group.append(False)
                                    else:
                                        tem_group.append(True)
                            if L_sub>0:
                                #如果上一帧与线叉积结果为正数，说明逆时针进入的，我们要看看原始的门方向是不是要计入逆时针进入
                                if gate_orientation2==1:
                                    tem_group.append(True)
                                else:
                                    #这里要做一个判断，因为是两线的门，所以你需要看看进入线是否为False，False才能append（False）进去
                                    #如果两线门的进入线已经为True了，你就不应该把它判负
                                    if not last_touch_list or (last_touch_list and last_touch_list[gate_point_ptr1][0]==False) or gate_point_ptr2!=0:
                                        tem_group.append(False)
                                    else:
                                        tem_group.append(True)
                            gate_point_ptr2+=1
                        else:
                            #这里要做一个判断，因为是两线的门，所以你需要看看进入线是否为False，False才能append（False）进去
                            #如果两线门的进入线已经为True了，你就不应该把它判负
                            if not last_touch_list or (last_touch_list and last_touch_list[gate_point_ptr1][0]==False) or gate_point_ptr2!=0:
                                tem_group.append(False)
                                gate_point_ptr2+=1
                            else:
                                tem_group.append(True)
                                gate_point_ptr2+=1

                    #上一帧在技术先或其延长线的情况
                    # elif L_sub==0:
                    #     #如果上一帧目标检测框中心点在计数线或其延长线的情况，判断上一帧中心点是否在计数线的延长线上
                    #     if self.on_sgement((x1,y1),(x2,y2),last_center_xy) and P_sub!=0:
                    #         #然后需要判断一下当前点的进入方向
                    #         if P_sub<0:
                    #             #如果当前点与线叉积结果为负数，说明逆时针进入的，我们要看看原始的门方向是不是要计入逆时针进入
                    #             if gate_orientation2==1:
                    #                 tem_group.append(True)
                    #             else:
                    #                 #这里要做一个判断，因为是两线的门，所以你需要看看进入线是否为False，False才能append（False）进去
                    #                 #如果两线门的进入线已经为True了，你就不应该把它判负
                    #                 if not last_touch_list or (last_touch_list and last_touch_list[gate_point_ptr1][0]==False) or gate_point_ptr2!=0:
                    #                     tem_group.append(False)
                    #                 else:
                    #                     tem_group.append(True)
                    #         if P_sub>0:
                    #             #如果当前点与线叉积结果为正数，说明顺时针进入的，我们要看看原始的门方向是不是要计入顺时针进入
                    #             if gate_orientation2==-1:
                    #                 tem_group.append(True)
                    #             else:
                    #                 #这里要做一个判断，因为是两线的门，所以你需要看看进入线是否为False，False才能append（False）进去
                    #                 #如果两线门的进入线已经为True了，你就不应该把它判负
                    #                 if not last_touch_list or (last_touch_list and last_touch_list[gate_point_ptr1][0]==False) or gate_point_ptr2!=0:
                    #                     tem_group.append(False)
                    #                 else:
                    #                     tem_group.append(True)
                    #         gate_point_ptr2+=1
                    #     else:
                    #         #这里要做一个判断，因为是两线的门，所以你需要看看进入线是否为False，False才能append（False）进去
                    #         #如果两线门的进入线已经为True了，你就不应该把它判负
                    #         if not last_touch_list or (last_touch_list and last_touch_list[gate_point_ptr1][0]==False) or gate_point_ptr2!=0:
                    #             tem_group.append(False)
                    #             gate_point_ptr2+=1
                    #         else:
                    #             tem_group.append(True)
                    #             gate_point_ptr2+=1

                    else:
                        #第一关没过，说明延长线都没有碰到，直接加一个FALSE到临时列表中
                        #这里要做一个判断，因为是两线的门，所以你需要看看进入线是否为False，False才能append（False）进去
                        #如果两线门的进入线已经为True了，你就不应该把它判负
                        if not last_touch_list or (last_touch_list and last_touch_list[gate_point_ptr1][0]==False) or gate_point_ptr2!=0:
                            tem_group.append(False)
                            gate_point_ptr2+=1
                        else:
                            tem_group.append(True)
                            gate_point_ptr2+=1

                        
                #for 循环结束，将临时列表加入到touch_list中
                touch_list.append(tem_group)
                gate_point_ptr1+=1
            #另一种情况是这个门只有一个标注线，反映在P_product和L_product上就是直接一个值
            elif isinstance(P_product,(int,float)) and isinstance(L_product,(int,float)):
                #获取标注线的元组
                gate_point3=gate_point_list[gate_point_ptr1]
                #获取标注线的四个坐标
                x1,y1,x2,y2=gate_point3
                #如果两个值的符号不同，说明过了第一关，可能碰到线也可能在延长线上，所以还要做第二个判断
                if P_product*L_product<0:
                    #第二关是要判断计数线的两个端点是否中心点连线的两侧
                    #拿到目标检测框中心点的四个坐标
                    a1,a2=xy_center_turple
                    b1,b2=last_center_xy
                    #现在要构造三个向量，首先要构造中心点连接向量AB
                    AB=(b1-a1,b2-a2)
                    #另外两个向量是计数线的两个端点 与xy_center_turple组成的向量
                    AP_1=(x1-a1,y1-a2)
                    AP_2=(x2-a1,y2-a2)
                    #然后现在分别计算AB与AP1，AB与AP2的叉积
                    cross_product1=AB[0]*AP_1[1]-AB[1]*AP_1[0]
                    cross_product2=AB[0]*AP_2[1]-AB[1]*AP_2[0]
                    #如果两个叉积的符号也是不同的，说明过了第二关，目标检测框的连线确实碰到了计数线
                    if cross_product1*cross_product2<0:
                        """
                        如果上一帧中心点叉积为负，当前中心点叉积为正，对应原始的方向应该是-1，就是顺时针方向进入
                        如果上一帧中心点叉积为正，当前中心点叉积为负，对应原始方向应该是1，是逆时针方向进入
                        """
                        if L_product<0 and P_product>0:
                            #这时候说明车是顺时针碰到的线，我们要看看原始门方向是不是-1
                            #如果是-1就说明是符合我们定义的方向进入的
                            if gate_orientation1==-1:
                                touch_list.append(True)
                            else:
                                touch_list.append(False)
                            gate_point_ptr1+=1
                        elif L_product>0 and P_product<0:
                            #这时候说明车是逆时针碰到的线，我们要看看原始门方向是不是1
                            #如果是1就说明是符合我们定义的方向进入的
                            if gate_orientation1==1:
                                touch_list.append(True)
                            else:
                                touch_list.append(False)
                            gate_point_ptr1+=1
                    else:
                        #没过第二关，说明连接线只是穿过延长线，加入false
                        touch_list.append(False)
                        gate_point_ptr1+=1
                
                #当前点落在计数线或其延长线的情况,在线上只用写一种如果再写上上一帧也在就会重复计数
                elif P_product==0.0:
                    #如果当前点落在计数线或其延长线的情况，判断当前点是否在计数线的延长线上
                    if on_segment((x1,y1),(x2,y2),xy_center_turple) and L_product!=0.0:
                        #判断一下当前点的进入方向
                        if L_product<0:
                            #如果上一帧与线叉积结果为负数，说明逆时针进入的，我们要看看原始的门方向是不是要计入逆时针进入
                            if gate_orientation1==-1:
                                touch_list.append(True)
                            else:
                                touch_list.append(False)
                        if L_product>0:
                            #如果上一帧与线叉积结果为正数，说明逆时针进入的，我们要看看原始的门方向是不是要计入逆时针进入
                            if gate_orientation1==1:
                                touch_list.append(True)
                            else:
                                touch_list.append(False)
                        gate_point_ptr1+=1
                    else:
                        #如果在计数线的延长线上，说明没有碰到计数线，直接将False加入临时列表中
                        touch_list.append(False)
                        gate_point_ptr1+=1

                #上一帧在技术先或其延长线的情况
                # elif abs(L_product)<1e-4:
                #     #如果上一帧目标检测框中心点在计数线或其延长线的情况，判断上一帧中心点是否在计数线的延长线上
                #     if self.on_sgement((x1,y1),(x2,y2),last_center_xy) and abs(P_product)>1e-4:
                #         #然后需要判断一下当前点的进入方向
                #         if P_product<0:
                #             #如果当前点与线叉积结果为负数，说明逆时针进入的，我们要看看原始的门方向是不是要计入逆时针进入
                #             if gate_orientation1==1:
                #                 touch_list.append(True)
                #             else:
                #                 touch_list.append(False)
                #         if P_product>0:
                #             #如果当前点与线叉积结果为正数，说明顺时针进入的，我们要看看原始的门方向是不是要计入顺时针进入
                #             if gate_orientation1==-1:
                #                 touch_list.append(True)
                #             else:
                #                 touch_list.append(False)
                #         gate_point_ptr1+=1
                #     else:
                #         #如果在计数线的延长线上，说明没有碰到计数线，直接将False加入临时列表中
                #         touch_list.append(False)
                #         gate_point_ptr1+=1
                
                else:
                    #第一关没有过说明没有碰到计数线或者计数线的延长线，直接给这个门的线赋值为False
                    touch_list.append(False)
                    gate_point_ptr1+=1

        return touch_list

def on_segment(a,b,p):
        """
        这个函数的功能是判断点p是否在向量ab上
        如果点p在向量ab上，返回True，否则返回False
        """
        return (
            min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and
            min(a[1], b[1]) <= p[1] <= max(a[1], b[1])
        )

#一个函数用于判断目标检测框的中心点在计数线的哪一侧
#以计数线靠左的点为轴（就是那一个x较小的点）。目标检测的中心如果在线的顺时针方向就为负，在逆时针方向为负
def judge_slide(xy_center_turple,gate_point_list):
    #获取目标检测框的中心点
    P_x,P_y=xy_center_turple
    #一个列表，按顺序记录目标检测框中心点与各个gate的叉积值
    cross_product_list=[]
    for item in gate_point_list:
        if not item:
            cross_product_list.append(None)
            continue
        #检擦第一个元素是否也是元组
        if len(item)>=2 and isinstance(item[0],(tuple,list)):
            temp_group=[]
            #情况1：包含多个元组的嵌套元组
            for sub_tuple in item:
                x1,y1,x2,y2=sub_tuple
                #构造两个叉积向量
                AB=(x2-x1,y2-y1)
                AP=(P_x-x1,P_y-y1)
                #计算叉积
                cross_product=AB[0]*AP[1]-AB[1]*AP[0]
                #将叉积加入到列表中
                temp_group.append(cross_product)
            cross_product_list.append(temp_group)
        elif len(item)==4 and all(isinstance(x, (int, float)) for x in item):
            #情况2：包含一个元组的情况
            x1,y1,x2,y2=item
            #构造两个叉积向量
            AB=(x2-x1,y2-y1)
            AP=(P_x-x1,P_y-y1)
            #计算叉积
            cross_product=AB[0]*AP[1]-AB[1]*AP[0]
            #将叉积加入到列表中
            cross_product_list.append(cross_product)
    return cross_product_list
    """
    这里返回的cross_product_list是一个列表,用于指示传入的目标检测框中心点，与json文件中所有线段的叉积值
    返回的列表形如：[[-1.11,2.22],3,-1,None]
    """


def get_point(data,index):
        #获取json文件中所有的线
        #一个列表用于存储所有的线
        line_list=[]
        line_list1=[]
        line_list2=[]
        gates=data["list"][index]["gate"]
        #循环提取所有line，并加入到line_list的列表中
        for gate in gates:
            #使用.get()方法，如果键不存在强制返回空列表 []
            line=gate.get("line",[])
            #同时做一个简单的类型判断，如果数据中存在乱七八糟的东西也强制添加空列表进去
            line_list.append(line if isinstance(line,list) else [])
            line1=gate.get("line1",[])
            line_list1.append(line1 if isinstance(line1,list) else [])
            line2=gate.get("line2",[])
            line_list2.append(line2 if isinstance(line2,list) else [])
        #一个列表用于记录所有的点
        point_list=[]
        point_list1=[]
        point_list2=[]
        index=0
        #第一个for循环得到的是一个门的所有线
        for gate_line in line_list:
            #判断一下，如果len(gate_line)>=2才接下一个for循环去取gate的一条线
            if len(gate_line)>=2:
                temp_group=[]
                #第二个for循环得到的是一个门的一条线
                for line in gate_line:
                    #获取线的两个端点
                    x1=line[0][0]
                    y1=line[0][1]
                    x2=line[1][0]
                    y2=line[1][1]
                    #将四个元素组合成元组传入临时列表
                    temp_group.append((x1,y1,x2,y2))
                point_list.append(tuple(temp_group))
                temp_group=[]
                #第二个for循环得到的是一个门的一条线
                for line in line_list1[index]:
                    #获取线的两个端点
                    x1=line[0][0]
                    y1=line[0][1]
                    x2=line[1][0]
                    y2=line[1][1]
                    #将四个元素组合成元组传入临时列表
                    temp_group.append((x1,y1,x2,y2))
                point_list1.append(tuple(temp_group))
                temp_group=[]
                #第二个for循环得到的是一个门的一条线
                for line in line_list2[index]:
                    #获取线的两个端点
                    x1=line[0][0]
                    y1=line[0][1]
                    x2=line[1][0]
                    y2=line[1][1]
                    #将四个元素组合成元组传入临时列表
                    temp_group.append((x1,y1,x2,y2))
                point_list2.append(tuple(temp_group))
                index+=1

            else:
                if len(gate_line)>=1:
                    #如果只有一个line就不用循环了直接 取
                    line=gate_line[0]
                    #获取线的两个端点
                    x1=line[0][0]
                    y1=line[0][1]
                    x2=line[1][0]
                    y2=line[1][1]
                    point_list.append((x1,y1,x2,y2))
                else:
                    point_list.append([])
                #如果只有一个line就不用循环了直接 取
                if index<len(line_list1) and len(line_list1[index])>=1:
                    line1=line_list1[index][0]
                    #获取线的两个端点
                    x1=line1[0][0]
                    y1=line1[0][1]
                    x2=line1[1][0]
                    y2=line1[1][1]
                    point_list1.append((x1,y1,x2,y2))
                else:
                    point_list1.append([])
                #如果只有一个line就不用循环了直接 取
                if index<len(line_list2) and len(line_list2[index])>=1:
                    line2=line_list2[index][0]
                    #获取线的两个端点
                    x1=line2[0][0]
                    y1=line2[0][1]
                    x2=line2[1][0]
                    y2=line2[1][1]
                    point_list2.append((x1,y1,x2,y2))
                else:
                    point_list2.append([])
                index+=1
        return (point_list,point_list1,point_list2)


def MakeGateLineJson(GateLineJsonPath,JsonFilePath):
    #初始化一个最后要写进GateLineJsonPath的数据
    GateLineJsonData={}
    #目标是获取point_list,gate_orientation，将它们写入一个单独的json文件中
    with open(JsonFilePath,"r",encoding="utf-8") as f:
        data = json.load(f)
        #获取json文件中那个大的列表
        cam_list = data["list"]
        #一个索引用于指向处理到第几个摄像头
        index=0
        #循环获取每一个设想头的信息，每一个信息都是字典
        for cam in cam_list:
            cam_name = cam["camera"]
            gate_orientation=cam.get("gate_orientation",[])
            point_list,point_list1,point_list2=get_point(data,index)
            #将point_list,point_list1,point_list2,gate_orientation写入GateLineJsonData
            GateLineJsonData[cam_name]={
                "point_list":point_list,
                "point_list1":point_list1,
                "point_list2":point_list2,
                "gate_orientation":gate_orientation
            }
            index+=1
    #将GateLineJsonData写入GateLineJsonPath
    with open(GateLineJsonPath,"w",encoding="utf-8") as f:
        json.dump(GateLineJsonData,f,ensure_ascii=False,indent=4)
    print("创建完成GateLineJsonPath文件")


def find_closest_time_index(time_entries, target_time_str):
    """在time_limit条目列表中找到最接近target_time_str的条目索引"""
    target = datetime.strptime(target_time_str, "%H:%M:%S")
    best_index = 0
    best_diff = float('inf')
    for i, entry in enumerate(time_entries):
        entry_time = datetime.strptime(entry["time_limit"], "%H:%M:%S")
        diff = abs((entry_time - target).total_seconds())
        if diff < best_diff:
            best_diff = diff
            best_index = i
    return best_index


def aggregate_count_csv(count_csv_path, num_gates):
    """读取_Count.csv，按gate_index和cls聚合，返回每个gate的分类计数字典列表"""
    cls_map = {0: "car", 1: "bus", 2: "truck", 3: "motorcycle"}
    #初始化每个gate的计数字典
    result = [{"car": 0, "bus": 0, "truck": 0, "motorcycle": 0} for _ in range(num_gates)]
    if not os.path.exists(count_csv_path):
        return result
    grouped = defaultdict(int)
    with open(count_csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                gate_idx = int(float(row["gate_index"]))
                cls_id = int(float(row["cls"]))
            except (KeyError, TypeError, ValueError):
                continue
            grouped[(gate_idx, cls_id)] += 1

    for (gate_idx, cls_id), count in grouped.items():
        gate_pos = int(gate_idx) - 1
        cls_name = cls_map.get(int(cls_id), None)
        if 0 <= gate_pos < num_gates and cls_name:
            result[gate_pos][cls_name] = int(count)
    return result


DEFAULT_ORI_JSON_PATH = str(PATHS.source_json_path)
DEFAULT_GATE_LINE_JSON_PATH = str(PATHS.gate_line_json_path)
DEFAULT_TRACKING_CSV_ROOT = str(PATHS.tracking_root)
DEFAULT_COUNT_CSV_ROOT = str(PATHS.count_root)
SEGMENT_NAME_RE = re.compile(r"^(?P<video>.+)_(?P<start>\d{2}_\d{2}_\d{2})__(?P<end>\d{2}_\d{2}_\d{2})$")
SUMMARY_HEADER = [
    "segment_name",
    "start_time",
    "end_time",
    "start_frame",
    "end_frame",
    "duration_sec",
    "gate_index",
    "car",
    "bus",
    "truck",
    "motorcycle",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Segment gate counting and camera summary generation.")
    parser.add_argument(
        "--csv-root",
        default=DEFAULT_TRACKING_CSV_ROOT,
        help="Root directory that contains per-camera tracking CSV folders.",
    )
    parser.add_argument(
        "--count-root",
        default=DEFAULT_COUNT_CSV_ROOT,
        help="Root directory for per-camera gate count CSV output.",
    )
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Optional segment_manifest.csv path. Defaults to <csv-root>/segment_manifest.csv.",
    )
    parser.add_argument(
        "--source-json",
        default=DEFAULT_ORI_JSON_PATH,
        help="Road definition JSON used to regenerate GateLineJson.",
    )
    parser.add_argument(
        "--gate-line-json",
        default=DEFAULT_GATE_LINE_JSON_PATH,
        help="Path to GateLineJson.json.",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=4,
        help="Number of worker processes for counting.",
    )
    parser.add_argument(
        "--skip-regenerate-gate-line",
        action="store_true",
        help="Skip regenerating GateLineJson.json if it already exists.",
    )
    return parser.parse_args()


def ensure_gate_line_json(gate_line_json_path, ori_json_path, skip_regenerate):
    gate_line_dir = os.path.dirname(gate_line_json_path)
    if gate_line_dir:
        os.makedirs(gate_line_dir, exist_ok=True)

    if (not skip_regenerate) or (not os.path.exists(gate_line_json_path)):
        MakeGateLineJson(gate_line_json_path, ori_json_path)

    with open(gate_line_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_cam_name_from_csv(csv_file):
    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    parts = base_name.split("_")
    if len(parts) < 2:
        return None

    cam_prefix = parts[0]
    cam_index = parts[1]
    if not cam_prefix:
        return None

    if not cam_index.isdigit():
        digits = "".join(ch for ch in cam_index if ch.isdigit())
        if not digits:
            return None
        cam_index = digits

    return f"cam_{cam_prefix}{cam_index}"


def resolve_manifest_path(csv_root_path, manifest_path):
    if manifest_path:
        return manifest_path
    return os.path.join(csv_root_path, "segment_manifest.csv")


def load_manifest_lookup(manifest_path):
    lookup = {}
    if not manifest_path or not os.path.exists(manifest_path):
        return lookup

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            segment_name = row.get("segment_name", "").strip()
            if segment_name:
                lookup[segment_name] = row
    return lookup


def parse_segment_name(segment_name):
    match = SEGMENT_NAME_RE.match(segment_name)
    if not match:
        return None
    return {
        "video_name": match.group("video"),
        "start_time": match.group("start").replace("_", ":"),
        "end_time": match.group("end").replace("_", ":"),
    }


def build_segment_tasks(csv_root_path, count_root_path, gate_line_json_data):
    tasks = []
    segment_infos = []
    skipped_completed = 0

    for dirpath, _, filenames in os.walk(csv_root_path):
        csv_files = sorted(
            f for f in filenames
            if f.endswith(".csv")
            and "_Count" not in f
            and not f.startswith("segment_manifest")
        )

        if not csv_files:
            continue

        video_name = os.path.basename(dirpath)
        for csv_file in csv_files:
            csv_path = os.path.join(dirpath, csv_file)
            base_name = os.path.splitext(csv_file)[0]
            cam_name = extract_cam_name_from_csv(csv_file)
            if not cam_name:
                print(f"Unexpected CSV name format, skip: {csv_file}")
                continue

            if cam_name not in gate_line_json_data:
                print(f"Camera {cam_name} not found in GateLineJson, skip: {csv_file}")
                continue

            cam_data = gate_line_json_data[cam_name]
            gate_orientation = cam_data["gate_orientation"]
            if not gate_orientation:
                print(f"Camera {cam_name} has empty gate_orientation, skip: {csv_file}")
                continue

            point_list = cam_data["point_list"]
            point_list1 = cam_data["point_list1"]
            point_list2 = cam_data["point_list2"]
            count_dir = os.path.join(count_root_path, video_name)
            count_csv_path = os.path.join(count_dir, f"{base_name}_Count.csv")
            done_flag_path = f"{count_csv_path}.ok"
            parsed_name = parse_segment_name(base_name) or {}

            info = {
                "video_name": video_name,
                "segment_name": base_name,
                "cam_name": cam_name,
                "csv_path": csv_path,
                "count_csv_path": count_csv_path,
                "num_gates": len(point_list),
                "start_time": parsed_name.get("start_time", ""),
                "end_time": parsed_name.get("end_time", ""),
            }
            segment_infos.append(info)

            if os.path.exists(count_csv_path) and os.path.exists(done_flag_path) and os.path.getsize(count_csv_path) > 34:
                skipped_completed += 1
                continue

            tasks.append((point_list, point_list1, point_list2, gate_orientation, csv_path, count_csv_path))

    return tasks, segment_infos, skipped_completed


def run_count_tasks(tasks, max_process_num):
    if not tasks:
        print("No pending count tasks. Skip counting phase.")
        return

    print(f"Total pending count tasks: {len(tasks)}")
    if max_process_num <= 1:
        for task in tasks:
            CountVechil(task)
        print("All count tasks finished.")
        return

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(max(1, max_process_num)) as pool:
        pool.map(CountVechil, tasks)
    print("All count tasks finished.")


def summary_sort_key(info, manifest_lookup):
    meta = manifest_lookup.get(info["segment_name"], {})
    start_frame = meta.get("start_frame", "")
    if str(start_frame).isdigit():
        return int(start_frame), info["segment_name"]
    return 10 ** 12, info["segment_name"]


def write_camera_summary(video_name, segment_infos, manifest_lookup, count_root_path):
    output_dir = os.path.join(count_root_path, video_name)
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, f"{video_name}_gate_summary.csv")

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_HEADER)
        writer.writeheader()

        for info in sorted(segment_infos, key=lambda item: summary_sort_key(item, manifest_lookup)):
            gate_counts = aggregate_count_csv(info["count_csv_path"], info["num_gates"])
            meta = manifest_lookup.get(info["segment_name"], {})
            start_time = meta.get("start_time", info["start_time"])
            end_time = meta.get("end_time", info["end_time"])
            start_frame = meta.get("start_frame", "")
            end_frame = meta.get("end_frame", "")
            duration_sec = meta.get("duration_sec", "")

            for gate_index, counts in enumerate(gate_counts, start=1):
                writer.writerow({
                    "segment_name": info["segment_name"],
                    "start_time": start_time,
                    "end_time": end_time,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "duration_sec": duration_sec,
                    "gate_index": gate_index,
                    "car": counts["car"],
                    "bus": counts["bus"],
                    "truck": counts["truck"],
                    "motorcycle": counts["motorcycle"],
                })

    print(f"Summary written: {summary_path}")


def main():
    args = parse_args()
    csv_root_path = args.csv_root
    count_root_path = args.count_root
    manifest_path = resolve_manifest_path(csv_root_path, args.manifest_path)

    if not os.path.isdir(csv_root_path):
        raise FileNotFoundError(f"CSV root path does not exist: {csv_root_path}")

    os.makedirs(count_root_path, exist_ok=True)
    gate_line_json_data = ensure_gate_line_json(
        args.gate_line_json,
        args.source_json,
        args.skip_regenerate_gate_line,
    )
    manifest_lookup = load_manifest_lookup(manifest_path)

    tasks, segment_infos, skipped_completed = build_segment_tasks(
        csv_root_path,
        count_root_path,
        gate_line_json_data,
    )
    print(
        f"Found {len(segment_infos)} valid tracking segments, "
        f"skip completed count files {skipped_completed}"
    )

    run_count_tasks(tasks, args.processes)

    grouped = defaultdict(list)
    for info in segment_infos:
        grouped[info["video_name"]].append(info)

    for video_name, infos in sorted(grouped.items()):
        write_camera_summary(video_name, infos, manifest_lookup, count_root_path)


if __name__ == '__main__':
    main()
