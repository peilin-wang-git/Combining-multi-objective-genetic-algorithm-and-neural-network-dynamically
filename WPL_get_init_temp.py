import pandas as pd
import os


def write_many(title, data):
    relative_location = "my_subject_kbann\\result21052203_everything_new_threshold50\\picture\\"
    name = relative_location + str(0) + "_" + title + ".csv"
#      print("name:",name)
    data.to_csv(name,index=True,sep=',')
    # write_log("----data "+str(0) + "_" + title + ".csv"+" has produced")

def get_value(file, line_num, range_up, range_down):
    file = open(file) 
    i = 0
    while 1:
        i += 1
        line = file.readline()
        # print("present:",line_num,i)
    # f = open(file)
    # text = f.read()
        if i==line_num:
            # print("***the whole text:",line)
            # print("***the text:",line[range_up:])
            print("***the text:",line[range_up:range_down])
            return float(line[range_up:range_down])
        if not line:
            print("not this")
            break
 
def compire_str(a,b):
    # print("compire:"+a+"###"+b+"###")
    if len(a)!=len(b):
        #print("False because len")
        return False
    else:
        for i in range(len(a)):
            #print(a[i],b[i])
            if a[i]!=b[i]:
                #print("False because str")
                return False
        return True


def get_modes(title,sample,columns,nmodes, mode_location):
    other_name = columns
    # samples = pd.DataFrame(columns=columns+output_name)
    location = mode_location + title + '\\'
    sub_files = os.listdir(location)
    # samples = pd.DataFrame()
    # mode = np.zeros(len(other_name)+len(output_name))
        
    # for i in range(len(other_name)):
        # if i<len(other_name):
            # mode[i] = sample[other_name[i]]
        
    for i in range(len(text_name)):        
        for sub_file in sub_files:
            # print("sub_file:\n",sub_file)
            sub_m = os.path.join(location,sub_file)
                
            if compire_str(text_name.loc[i,"name"]+".txt", sub_file):
                if nmodes==1:
                    sample["Mode1"+text_name.loc[i,"reflect_name"]] = get_value(sub_m, text_name.loc[i,"line"], text_name.loc[i,"range_down"], text_name.loc[i,"range_up"])
                else:
                    sample["Mode2"+text_name.loc[i,"reflect_name"]] = get_value(sub_m, text_name.loc[i,"line"], text_name.loc[i,"range_down"], text_name.loc[i,"range_up"])

    # samples = samples.append(sample)
    # print("samples\n",sample)
      
    return sample


calculate_num = 3
text_name =  pd.DataFrame([["Mode1Frequency",3,30,-1,"_frequent"],
                                        ["Mode1R_Q",3,30,-1,"_R_divide_Q"],
                                        ["Mode1ShuntImpedance",3,30,-1,"_shunt_impedance"],
                                        ["Mode1Q-Factor",3,30,-1,"_Q-factor"],
                                        ["Mode_1_Coffs",9,0,-1,"_E_divide_H"]],columns=["name","line", "range_down","range_up","reflect_name"])
relative_location = "my_subject_kbann\\result21052203_everything_new_threshold50\\picture\\"
input_name = ['Req', 'Leq', 'R3_left', 'R3_right', 'cocave', 'concave_L', 'nose', 'r0_left', 'r0_right', 'r1_left', 'r1_right', 'r2_left', 'r2_right']
hidden_name = ["random1","random2","random3_r","random3_l","random4","random5_r","random5_l","random6_r","random6_l","random7_r","random7_l","random8_r","random8_l"]
mode_location = "my_subject_kbann\\result21052203_everything_new_threshold50\\part1\\"
samples_get_result = pd.read_csv(relative_location + "0_samples_get_result_3.csv")
samples_get_result.index = samples_get_result["Unnamed: 0"]
samples_get_result = samples_get_result.drop(["Unnamed: 0"],1)
number_list = list(samples_get_result[samples_get_result["Mode2_E_divide_H"]==0].index)
samples_get_result_temp = pd.DataFrame()
for i in number_list:
    print(i)
    sample = get_modes("000000_NEW_Mode"+str(100+calculate_num)[1:]+"_"+str(i), samples_get_result.loc[i, input_name+hidden_name], hidden_name+input_name, calculate_num, mode_location)
    samples_get_result_temp = samples_get_result_temp.append(sample)
write_many("samples_get_result_temp_"+str(calculate_num),samples_get_result_temp)