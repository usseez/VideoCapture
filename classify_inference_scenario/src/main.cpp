//input : classified video based on scenario
//output : captured image based on scenario
//이미 분류된 demo video를 바탕으로 시나리오에 적합한 구간이라고 생각하면 c를 눌렀을때부터, e 눌렀을때까지 캡처!!
//key input
// c: start capture
// v : end capture
// n : go to the next video
#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "../include/cut_save.h"
#include <dirent.h>
#include <sys/stat.h>
char origin_video_folder_path[128] = {"/home/ubuntu/Dataset/00.scenario_video_classify/20251219"};
char new_video_folder_path[MAX_LENGTH] = {"/home/ubuntu/Dataset/00.scenario_v1/demo_251219"};
bool clicked = false;
int clicked_frame = 0;
int total_frames;
int frame_width;
int frame_height;
int bar_height = 60;
float scale = 0.2f;
int start_frame = 0;
int end_frame = 0;

void onMouse(int event, int x, int y, int flags, void* userData){//frame 이동 위해
    float ratio = 0.0f;
    int x_original = (int)(x / scale);
    int y_original = (int)(y / scale);
    if(event == cv::EVENT_LBUTTONDOWN &&
        y_original >= frame_height && y_original < frame_height + bar_height){

        ratio = (float)x_original / frame_width;
        clicked_frame = (int)(ratio * total_frames);
        clicked = true;
    }

}

void captureFile(const char *folderpath){

    struct dirent *entry;
    struct dirent *subentry;
    char scenario_foler_path[256];
    char file_path[512];
    bool playVideo = true;
    char output_video_path[256];
    int cap_number = 0;
    char output_video_file[512];
    char video_capture_file[512];
    char video_file_name[512];
    float progress = 0.0;
    int current_frame = 0;
    int bar_width;
    bool capturing = false;
    
    DIR *dir = opendir(folderpath);
    if(dir == nullptr){
        printf("cannot open folder: %s", folderpath);
        return;
    }
    else{
        printf("open folder: %s\n", folderpath);
    }


    mkdir(new_video_folder_path, 0757);
    


    while((entry = readdir(dir)) != nullptr){
        if (entry->d_type == DT_DIR && strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0){
            sprintf(scenario_foler_path, "%s/%s", origin_video_folder_path, entry->d_name);
            
            DIR *subdir = opendir(scenario_foler_path);
            if(subdir == nullptr){
                printf("[Cannot open folder]: %s\n", scenario_foler_path);
                return;
            }
            else{
                printf("[Open folder]: %s\n", scenario_foler_path);
                //show and cut
                while((subentry = readdir(subdir)) != nullptr){
                    if (subentry->d_type == DT_REG) { //(*subentry).d_type = subentry->d_type
                        const char *filename = subentry->d_name;
                        bool playing = true;
                        sprintf(file_path, "%s/%s", scenario_foler_path, filename);
                        sprintf(output_video_path, "%s/%s", new_video_folder_path, entry->d_name);
                        
                        mkdir(output_video_path, 0755);
                        
                        sprintf(output_video_file, "%s/%s", output_video_path, filename);
                        cv::VideoCapture cap(file_path);
                        
                        if(!cap.isOpened()){
                            printf("[Can't Open the video in {%s}]\n", scenario_foler_path);
                        }

                        cap_number = 0;
                        total_frames = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
                        frame_height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
                        frame_width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);

                        cv::Mat frame, resized;
                        capturing = false;
                        
                        cv::namedWindow(entry->d_name);
                        cv::setMouseCallback(entry->d_name, onMouse);
                        
                        cap_number = 0;
                        while(playing){

                            int key = cv::waitKey(10);

                            cap >> frame; //예외처리하는 함수사용하기?
                            current_frame = (int)cap.get(cv::CAP_PROP_POS_FRAMES);
                            if(frame.empty()){
                                playing = false;
                                printf("empty image\n");
                                continue;
                            }
                            
                            
                            //show progress bar
                            cv::Mat display;
                            cv::copyMakeBorder(frame, display, 0, 60, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(10, 10, 10));
                            progress = (float)current_frame / total_frames;
                            bar_width = (int)(progress * frame.cols); //?
                            cv::rectangle(display, cv::Point(0, frame.rows), cv::Point(bar_width, frame.rows + bar_height), cv::Scalar(255, 255, 255), cv::FILLED);
                            
                            //show the video
                            cv::resize(display, resized, cv::Size(), 0.2, 0.2);
                            cv::imshow(entry->d_name, resized);
                            
                            //mouse event
                            if(clicked){
                                cap.set(cv::CAP_PROP_POS_FRAMES, clicked_frame);
                                clicked = false;
                            }
                            
                            //key input process
                            if(key == 27){ //esc
                                cap.release();
                                cv::destroyAllWindows();
                                closedir(subdir);
                                closedir(dir);
                                return;     //quit
                            }
                            if(key == 32){      //space bar : pause
                                cv::waitKey(-1);
                            }
                            else if(key == 'c'){    //set capturemode
                                start_frame = (int)cap.get(cv::CAP_PROP_POS_FRAMES);
                                cap.set(cv::CAP_PROP_POS_FRAMES, start_frame);
                                capturing = true;
                                printf("starting capture...\n");
                            }
                            else if(key == 'v'){    //end capturemode
                                capturing = false;
                                printf("end capture...\n");
                            }
                            else if(key == 'n'){    //go to the next video
                                break;
                            }
                                
                            
                            if(capturing){
                                sprintf(video_file_name, output_video_file);
                                video_file_name[strlen(output_video_file) - 4] = '\0';
                                sprintf(video_capture_file, "%s_%04d.png", video_file_name, current_frame);
                                printf("[%d]_%s\n", cap_number, video_capture_file);
                                cv::imwrite(video_capture_file, frame);
                                cap_number++;

                                if(!capturing){
                                    playing = false;
                                    cap.release();
                                    continue;
                                }
                            }

                        }
                        cap.release();
                        cv::destroyAllWindows();
                    }
                }
                closedir(subdir);
            }
        }
    }
    closedir(dir);

}



int main(void)
{
    captureFile(origin_video_folder_path);

    return 0;

}
