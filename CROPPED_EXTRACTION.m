clc;
clear all;
close all;

start_path = 'E:\DATASET';                                   % define a starting folder

topLevelFolder = uigetdir(start_path , 'Mammogram Dataset(TITLE)');
if topLevelFolder == 0
    fprintf('no choice of folder was made')
end

% Get list of all subfolders
allSubFolders = genpath(topLevelFolder);

SubFolders = allSubFolders;                                 % parse into a cell array
MC_ROIfoldersList = {};
MC_ROIimages = 0;     

while true
    
  % scans str and if it meets a delimiter it separates str in a and b 
  [singleSubFolder, SubFolders] = strtok(SubFolders, ';');
  
  if isempty(singleSubFolder)   
    break;
  end
  
  % searches for occurrences of str and returns the starting index of every instance 
  MCindex = strfind( singleSubFolder, 'Calc');
    
  if isempty(MCindex) == 0
    MCFULLindex = strfind( singleSubFolder , 'ROI mask images' );
    if isempty(MCFULLindex) == 0      
        MC_ROIfoldersList = [MC_ROIfoldersList singleSubFolder];
    end
  end
end

MC_ROIfoldersNumber = length(MC_ROIfoldersList);

for j = 1 : MC_ROIfoldersNumber       % loop of subfolders
    
  % Get this folder  
  current1 = MC_ROIfoldersList{j};
  [a,b] = strtok(current1, '-');
  [a,b] = strtok(b, '\');
  [final,b] = strtok(b, '\');
  
  % Get DCM files  
  filePattern1 = sprintf('%s\\000001.dcm', current1);       % formats the data in arrays and returns the resulting text
  baseFileNames1 = dir(filePattern1);                       % returns attributes 
   
  numberOfMC_ROIFiles = length(baseFileNames1);             % number of images in subfolder  
  
  if numberOfMC_ROIFiles >= 1                                 
    for f = 1 : numberOfMC_ROIFiles
        MC_ROIimages = MC_ROIimages + 1;      
        imageFP = fullfile(current1, baseFileNames1(f).name);
        newfolder = 'E:\PROJECT\MC_MASKS';                         
        status = copyfile(imageFP, newfolder);
        if status == 11
            fprintf('Image moved from %s \n', imageFP);
        end
        
        renamed = sprintf(final);
        
        movefile( fullfile(newfolder,baseFileNames1(f).name) , fullfile(newfolder,final) );
        
        outputBaseFileName = sprintf('%s.dcm', renamed);
        movefile( fullfile(newfolder,renamed) , fullfile(newfolder,outputBaseFileName) )        
        
    end
  else
    fprintf('Folder %s has no image files in it.\n', current1);
  end
end
