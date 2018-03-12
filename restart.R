getwd()
setwd("/users/mis/mohafnu/restarts")
##################
# Load data file #
##################
set.seed(111)
churn0 <- read.csv("/users/mis/mohafnu/restarts/non_restarted.csv", sep=",", header= FALSE, na.strings='NULL')
churn1 <- read.csv("/users/mis/mohafnu/restarts/restarted.csv", sep=",", header= FALSE, na.strings='NULL')
colnames(churn0 ) <- as.character(unlist(churn0 [1,]))
colnames(churn1 ) <- as.character(unlist(churn1 [1,]))
churn0$acct_no <- format(churn0$acct_no, scientific = FALSE)
churn1$acct_no <- format(churn1$acct_no, scientific = FALSE)
churn0 <- churn0[-1, ]
churn1 <- churn1[-1, ]
churn0 <- churn0[ which(churn0$disco_type=='DV' | churn0$disco_type=='PIADV' | churn0$disco_type=='PIANP' | churn0$disco_type=='NP'), ]
churn1 <- churn1[ which(churn1$disco_type=='DV' | churn1$disco_type=='PIADV' | churn1$disco_type=='PIANP' | churn1$disco_type=='NP'), ]

#colnames(churn0) <- tolower(colnames(churn0))
#colnames(churn1) <- tolower(colnames(churn1))
#colnames(churn0) <- sub("dish360_restarts_ac[.]", "", colnames(churn0))
#colnames(churn1) <- sub("dish360_restarts_vd[.]", "", colnames(churn1))
churn0_basic <- churn0[sample(1:nrow(churn0), 50000,replace=FALSE),]
churn1_basic <- churn1[sample(1:nrow(churn1), 50000,replace=FALSE),]
churn0_valid <- churn0[ !(churn0$acct_no %in% churn0_basic$acct_no), ]
churn1_valid <- churn1[ !(churn1$acct_no %in% churn1_basic$acct_no), ]


churn0_basic$is_restart <- 0
churn1_basic$is_restart <- 1
churn0_valid$is_restart <- 0
churn1_valid$is_restart <- 1
train <- rbind(churn0_basic,churn1_basic)
score <- rbind(churn0_valid,churn1_valid)
write.csv(score,file ="/users/mis/mohafnu/restarts/score.csv")
##################
# library paths  #
##################

.libPaths('/users/mis/mohafnu/R/x86_64-redhat-linux-gnu-library/3.1')

####################
# Common Libraries #
####################
library(gains)
library(plyr)
library(MASS)
library(boot)
library(survey)
library(mitools)
library(caTools)
library(ROCR)
library(circular)
library(wle)
library(leaps)
library(locfit)
library(rJava)
library(coin)
library(modeltools)
library(vcd)
library(grid)
library(dummies)
library(caret)
library(RRF)
library(class)
library(Matrix)
library(randomForest)
library(pROC)
library(mlbench)
library(adabag)
library(rpart)
library(doParallel)
library(foreach)
library(ipred)
library(chron)
library(sqldf)
library(doMC)
library(zoo)
library(e1071)
registerDoMC(16)   #run 25 trees on each of 16 cores
getDoParWorkers()
library(SDMTools)  #accuracy with threshold level




#########################################################################
# Convert english programming categorical variable into dummy variables #
#########################################################################
#Drop <-  names(train[, sapply(train, function(v) var(v, na.rm=TRUE)==0)])


Drop <- c("new_return_type","ib_ra","ib_model_num","ra_receipt_date","ib_location","x_cust_type","ib_tracking","pay_terms","detail_line_type","part_price","sys","grocery_cc","sporting_goods_cc","truck_owner","new_car_purch","motorcycle_own","recreat_own","bb_home","hopper","analog_cable_sub_count","analog_cableover_sub_count","analog_cableprim_sub_count","digcable_sub_count","digcableover_sub_count","digcableprim_sub_count","insight_cable_sub_count","paytv_pen","cable_pen","dbs_pen","dish_pen","directv_pen","fibertv_pen","cableover_pen","cableprim_pen","analog_cable_pen","analog_cableover_pen","analog_cableprim_pen","digcable_pen","digcableover_pen","digcableprim_pen","hdtv_pen","broadband_pen","dsl_pen","cablemodem_pen","fiostv_pen","uverse_pen","fiosdata_pen","uversedata_pen","qwestdata_pen","digcable_hh_count","insight_cable_hh_count","qwest_telco_hh_count","embarq_telco_hh_count","windstream_telco_hh_count","centurytel_telco_hh_count","frontier_telco_hh_count","zip_cns_blk_gp_curr_ind","cnss_blk_grp_start_date","zip_end_date","zip_start_date","hist_end_date","zip_end_date","account_num","ib_ra","ib_model_num","ra_receipt_date","ra_ship_date","ra_create_date","ra_receipt_date","ib_location","ra_creation_date","ra_row","discodate","serial_no","sc_no","ra_no","last_update_date","equip_status","ob_ra","ob_model_num","header_creator","cust_failure_code","cust_sec_fc","family","ob_tracking","serial_no2","discodate2","discoid2","dish_account_id","primary_receiver_id","imputed","prin","acct_num","fiostv_sub_count","qwestdata_sub_count","other_broadband_sub_count","train$comcast_cable_sub_count","time_warner_cable_sub_count","train$charter_cable_sub_count","cox_cable_sub_count","cablevision_cable_sub_count","bright_house_cable_sub_count","suddenlink_cable_sub_count","mediacom_cable_sub_count","cable_one_cable_sub_count","tv_pen","not_cabled_hh_count","cabled_hh_count","fibertv_hh_count","fiostv_hh_count","uverse_hh_count","fiosdata_hh_count","uversedata_hh_count","qwestdata_hh_count","broadband_hh_count","dsl_hh_count","cablemodem_hh_count","comcast_cable_hh_count","time_warner_cable_hh_count","charter_cable_hh_count","cox_cable_hh_count","cablevision_cable_hh_count","bright_house_cable_hh_count","suddenlink_cable_hh_count","mediacom_cable_hh_count","cable_one_cable_hh_count","att_telco_hh_count","verizon_telco_hh_count")
train <- train[,!(names(train) %in% Drop)]

#########################
# zero variance dropped #
#########################
################# Check datatype
#Type.Matrix = as.data.frame(NULL)
#for(i in 1:ncol(train))
#{
 # Variable.Type = class(train[,i])
 # Variable.Name = names(train)[i]
  
 # Type = cbind(Variable.Name,Variable.Type)
 # Type.Matrix = rbind(Type.Matrix,Type)
#}
#print(Type.Matrix)

####################################
# dependent variable rearrangement #
####################################


train <- train[c(301,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,289,290,291,292,293,294,295,296,297,298,299,300)]



########################
# Data type correction #
######################## 

trim <- function (x) gsub("^\\s+|\\s+$", "", x)
train$adults_65_74 <- as.factor(	trim(train$adults_65_74)	)
train$adults_55_64 <- as.factor(	trim(train$adults_55_64)	)
train$hhold_age	<-	as.factor(	trim(train$hhold_age)	)
train$ra_age <- format(train$ra_age, scientific = TRUE)
train$ra_age	<-	as.numeric(	train$ra_age	)
train$tenure <- format(train$tenure, scientific = TRUE)
train$tenure	<-	as.numeric(	train$tenure	)
train$credit_score <- format(train$credit_score, scientific = TRUE)
train$credit_score	<-	as.numeric(	train$credit_score	)
train$no_of_receivers <- format(train$no_of_receivers, scientific = TRUE)
train$no_of_receivers	<-	as.numeric(	train$no_of_receivers	)
train$loyalty_call_count <- format(train$loyalty_call_count, scientific = TRUE)
train$loyalty_call_count	<-	as.numeric(	train$loyalty_call_count	)
train$count_of_latecharge <- format(train$count_of_latecharge, scientific = FALSE)
train$count_of_latecharge	<-	as.factor(	trim(train$count_of_latecharge)	)
train$mst_rcnt_inqry_age_num <- format(train$mst_rcnt_inqry_age_num, scientific = TRUE)
train$mst_rcnt_inqry_age_num	<-	as.numeric(	train$mst_rcnt_inqry_age_num	)
train$aggr_amt_derog_pub_rcnt_36_mos <- format(train$aggr_amt_derog_pub_rcnt_36_mos, scientific = TRUE)
train$aggr_amt_derog_pub_rcnt_36_mos	<-	as.numeric(	train$aggr_amt_derog_pub_rcnt_36_mos	)
train$satsfctry_trds_opn_24_mos_num <- format(train$satsfctry_trds_opn_24_mos_num, scientific = TRUE)
train$satsfctry_trds_opn_24_mos_num	<-	as.numeric(	train$satsfctry_trds_opn_24_mos_num	)
train$trds_gt_50_lmt_pcntg <- format(train$trds_gt_50_lmt_pcntg, scientific = TRUE)
train$trds_gt_50_lmt_pcntg	<-	as.numeric(	train$trds_gt_50_lmt_pcntg	)
train$snc_rcnt_delnqncy_mos <- format(train$snc_rcnt_delnqncy_mos, scientific = TRUE)
train$snc_rcnt_delnqncy_mos	<-	as.numeric(	train$snc_rcnt_delnqncy_mos	)
train$snc_oldst_bankcard_trd_opn_mos <- format(train$snc_oldst_bankcard_trd_opn_mos, scientific = TRUE)
train$snc_oldst_bankcard_trd_opn_mos	<-	as.numeric(	train$snc_oldst_bankcard_trd_opn_mos	)
train$snc_rcnt_bankcard_trd_opn_mos <- format(train$snc_rcnt_bankcard_trd_opn_mos, scientific = TRUE)
train$snc_rcnt_bankcard_trd_opn_mos	<-	as.numeric(	train$snc_rcnt_bankcard_trd_opn_mos	)
train$bnk_crd_trds_gt_75_lmt_pcntg <- format(train$bnk_crd_trds_gt_75_lmt_pcntg, scientific = TRUE)
train$bnk_crd_trds_gt_75_lmt_pcntg	<-	as.numeric(	train$bnk_crd_trds_gt_75_lmt_pcntg	)
train$bnk_revlvng_trds_num <- format(train$bnk_revlvng_trds_num, scientific = TRUE)
train$bnk_revlvng_trds_num	<-	as.numeric(	train$bnk_revlvng_trds_num	)
train$tot_bnk_revlvng_hi_crdt_amt <- format(train$tot_bnk_revlvng_hi_crdt_amt, scientific = TRUE)
train$tot_bnk_revlvng_hi_crdt_amt	<-	as.numeric(	train$tot_bnk_revlvng_hi_crdt_amt	)
train$trds_mop_equal_2_or_gt_num <- format(train$trds_mop_equal_2_or_gt_num, scientific = TRUE)
train$trds_mop_equal_2_or_gt_num	<-	as.numeric(	train$trds_mop_equal_2_or_gt_num	)
train$dept_stre_trds_num <- format(train$dept_stre_trds_num, scientific = TRUE)
train$dept_stre_trds_num	<-	as.numeric(	train$dept_stre_trds_num	)
train$satsfctry_dept_stre_trds_num <- format(train$satsfctry_dept_stre_trds_num, scientific = TRUE)
train$satsfctry_dept_stre_trds_num	<-	as.numeric(	train$satsfctry_dept_stre_trds_num	)
train$fin_trds_lmt_gt_50_pcntg <- format(train$fin_trds_lmt_gt_50_pcntg, scientific = TRUE)
train$fin_trds_lmt_gt_50_pcntg	<-	as.numeric(	train$fin_trds_lmt_gt_50_pcntg	)
train$highst_delnqncy_ever_on_trdnum <- format(train$highst_delnqncy_ever_on_trdnum, scientific = TRUE)
train$highst_delnqncy_ever_on_trdnum	<-	as.numeric(	train$highst_delnqncy_ever_on_trdnum	)
train$tot_amt_now_past_due_amt <- format(train$tot_amt_now_past_due_amt, scientific = TRUE)
train$tot_amt_now_past_due_amt	<-	as.numeric(	train$tot_amt_now_past_due_amt	)
train$mos_on_file_num <- format(train$mos_on_file_num, scientific = TRUE)
train$mos_on_file_num	<-	as.numeric(	train$mos_on_file_num	)
train$crrntly_actv_mortgg_trds_num <- format(train$crrntly_actv_mortgg_trds_num, scientific = TRUE)
train$crrntly_actv_mortgg_trds_num	<-	as.numeric(	train$crrntly_actv_mortgg_trds_num	)
train$mos_snc_oldst_mortgg_opnd_num <- format(train$mos_snc_oldst_mortgg_opnd_num, scientific = TRUE)
train$mos_snc_oldst_mortgg_opnd_num	<-	as.numeric(	train$mos_snc_oldst_mortgg_opnd_num	)
train$psnl_fin_inqurs_in_24_mos_num <- format(train$psnl_fin_inqurs_in_24_mos_num, scientific = TRUE)
train$psnl_fin_inqurs_in_24_mos_num	<-	as.numeric(	train$psnl_fin_inqurs_in_24_mos_num	)
train$snc_oldst_retl_trd_opn_mos_num <- format(train$snc_oldst_retl_trd_opn_mos_num, scientific = TRUE)
train$snc_oldst_retl_trd_opn_mos_num	<-	as.numeric(	train$snc_oldst_retl_trd_opn_mos_num	)
train$opn_trds_num <- format(train$opn_trds_num, scientific = TRUE)
train$opn_trds_num	<-	as.numeric(	train$opn_trds_num	)
train$tv_hh_count <- format(train$tv_hh_count, scientific = TRUE)
train$tv_hh_count	<-	as.numeric(	train$tv_hh_count	)
train$broadband_sub_count <- format(train$broadband_sub_count, scientific = TRUE)
train$broadband_sub_count	<-	as.numeric(	train$broadband_sub_count	)
train$legacy_unit <- as.factor((trim(train$legacy_unit)))

#################
# replace missing values
# numeric columns

train[is.na(train)] <- 0

# factor columns
for(i in 1:ncol(train))
{
  if(is.factor(train[,i]))
  {
    train[,i] = as.character(train[,i])
    train[,i][train[,i]=="NULL"] = "UNKNOWN"
    train[,i] = as.factor(train[,i])
  }
}

for(i in 1:ncol(train))
{
  if(is.factor(train[,i]))
  {
    train[,i] = as.character(train[,i])
    train[,i][train[,i]=="null"] = "UNKNOWN"
    train[,i] = as.factor(train[,i])
  }
}

for(i in 1:ncol(train))
{
  if(is.factor(train[,i]))
  {
    train[,i] = as.character(train[,i])
    train[,i][train[,i]=="NA"] = "UNKNOWN"
    train[,i] = as.factor(train[,i])
  }
}

for(i in 1:ncol(train))
{
  if(is.factor(train[,i]))
  {
    train[,i] = as.character(train[,i])
    train[,i][train[,i]==""] = "UNKNOWN"
    train[,i] = as.factor(train[,i])
  }
}

for(i in 1:ncol(train))
{
  if(is.factor(train[,i]))
  {
    train[,i] = as.character(train[,i])
    train[,i][train[,i]=="<NA>"] = "UNKNOWN"
    train[,i] = as.factor(train[,i])
  }
}

##############################
# Screening Factor Variables #
##############################

attach(train)
strt = 3
end = ncol(train)
var = names(train)
screening_table_fact = data.frame(NULL)
for(i in strt:ncol(train)){
  
  if (is.factor(train[,i])){
    
    table1=table(train[,i],train[,1])
    table2=as.data.frame(prop.table(table1,2))
    colnames(table2) = c(var[i],var[1],"Freq")
    
    table20 = table2[table2$is_restart==0,]
    colnames(table20)=c(var[i],var[1],"Roll")
    
    table21=table2[table2$is_restart==1,]
    colnames(table21)=c(var[i],var[1],"Resolved")
    
    table201 = merge(table21[,-2],table20[,-2],by=var[i])
    
    attach(table201)
    for(j in 1:nrow(table201)){### Weight of Evidence
      if(Roll[j] > 0 & Resolved[j]>0)
      { 
        table201$WOE[j] = round((Resolved[j]-Roll[j])*log(Resolved[j]/Roll[j]),digits=4)
      } 
      else
      {
        table201$WOE[j] = 0 
      } 
    }
    Entropy = sum(table201$WOE)  #### Information Value Calculation
    detach(table201)
    
    variable = factor(names(train))[i]
    ChiSquare = chisq.test(train[,i],train[,1])[1] ### Pearson's Chi-Square
    pvalue = chisq.test(train[,i],train[,1])[3]    ### Pearson's Chi-Square p value
    
    Screening=data.frame(variable,ChiSquare,pvalue,Entropy)
    screening_table_fact = data.frame(rbind(screening_table_fact,Screening))
    
  }
}
print(screening_table_fact)
write.csv(screening_table_fact, file ="/users/mis/mohafnu/restarts/screening_table_fact.csv")



Drop <- c("sales_call_count","hbo_current","tailgater","online_hhold","dish_referrer","adults_over_65_infr","starz_current","adults_45_64_infr","fashion","stamp_collectibles","adults_18_24","hist_start_date","donor_political_liberal","adults_no_age","adults_under_35_infr","books_reading","bbhome_current","adults_35_44_infr","encore_current","home_owner","coin_collectibles","cnss_blk_grp_cd","dish_referree","electronics","child_0_2","traveler","lang","cinemax_ever","ppv_buys_6months","african_amer_conf","nascar","donate_to_charitable_causes","cinemax_current","all_hobbies","any_cc","cooking","mail_order","sports","child_3_5","motorcycle_riding","interior_decorating","marital_status","bird_feeding_watching","child_16_17","standard_retail_credit_card","showtime_current","biz_travel","core_latino","adults_25_34","overbuilder_cable_sp_name","dwelling_type","art_antique_collectibles","mail_order_buyer","percnt_in","fitness_exercise","any_collectibles","self_improve","mail_order_donor","donor_political_conservative","contests_sweepstakes","gardening","casino_gambling","bank_cc","leisure_travel","oil_and_gas_cc","international_travel","child_6_10","hhold_type","sewing_needlework_knitting","skiing_snowboarding","education_years","child_11_15","ethnic_group","wildlife_environmental_causes","mail_order_dvd","misc_credit_card","home_improvement","adults_35_44","adults_over_75","automotive_work","cat","pets","crafts","cruise_ship_vacation","dishnet","gourmet_foods","running_jogging","walking","weight_control","wines","dog","adults_45_54","fishing","cycling","advantage_number_of_adults","photo","upscale_retail_specialty_cc","grandkids","natural_foods","science","science_new_technology","camping_hiking","bible","hbo_ever","art_event","hunting_shooting","finance_company_cc","starz_ever","showtime_ever","cnty_fips_cd","needs_based_segment","terr_fips_cd","state_code","local_cable_sp_name","commitment_end_date","fips_st_co","zip_4_code","cnss_tract_cd","original_acct_activation_date","last_disconnect_date","cbg","centrics_zip_code","zip_5_code","stars","account_status_code","first_activation_date","income","equip_status","equip_status_code","tenure_days","new_return_type","cust_zip_code","ob_ra","ib_ra","ob_model_num","ib_model_num","ra_ship_date","ra_create_date","ra_receipt_date","header_creator","ob_location","ib_location","ship_location","line_creator","account_num","x_cust_type","cust_failure_code","cust_sec_fc","part_number","family","line","ib_tracking","ob_tracking","warranty_type","pay_terms","detail_line_type","cust_location","part_price","serial_no2","disposition","invalid_ra_flag","discodate2","discoid2","beacon_score_bucket","core_english","primary_cable_sp_name","primary_telco_sp_name","prime_post_office_name")

train <- train[,!(names(train) %in% Drop)]

##############################
# Screening Numeric Variables #
##############################

attach(train)
Screening.Table <- data.frame(NULL)
#### Screening.Numeric = data.frame(NULL)
for(i in 3:ncol(train)){ 
  if(is.numeric(train[,i])){
    
    interim = train[,c(1,i)]
    interim$RANK = cut(interim[,2],unique(quantile(interim[,2],(0:15)/15,type=3,ties=low,na.rm=TRUE)),include.lowest=TRUE)
    
    
    table1=table(interim[,2],interim[,1]) ### Contingency Table
    table2=as.data.frame(prop.table(table1,2)) ### Proportion Table
    names(table2)[(which(names(table2)==c("Var1","Var2")))] <- c(names(interim)[3],names(interim)[1]) 
    
    table20 = table2[table2$is_restart==0,]
    names(table20)[(which(names(table20)==c("Freq")))] <- c("Roll")
    
    table21=table2[table2$is_restart==1,]
    names(table21)[(which(names(table21)==c("Freq")))] <- c("Resolved")
    
    table201 = merge(table21[,-2],table20[,-2],by="RANK")
    
    attach(table201)
    for(j in 1:nrow(table201)){### Weight of Evidence
      if(Roll[j] > 0 & Resolved[j]>0)
      { 
        table201$WOE[j] = round((Resolved[j]-Roll[j])*log(Resolved[j]/Roll[j]),digits=4)
      } 
      else
      {
        table201$WOE[j] = 0 
      }
    }
    Entropy = sum(table201$WOE)  #### Information Value Calculation
    detach(table201)
    
    
    ChiSquare <- chisq.test(interim[,3], interim[,1])[1]  #### Pearson's Chi-Square
    pvalue <- chisq.test(interim[,3], interim[,1])[3]     #### Pearson's Chi-Square p value
    
    variable <- factor(names(interim))[2]
    
    Screening=data.frame(variable,ChiSquare,pvalue,Entropy)
    
    Screening.Table = data.frame(rbind(Screening.Table,Screening))
    
  }
}
print(Screening.Table)

detach(train)
write.csv(Screening.Table, file ="/users/mis/mohafnu/restarts/screening_table_num.csv")

Drop <- c("trds_max_delnqncy_2_in_24_mos","crrntly_actv_psnl_fin_trds_num","trds_wth_histmop_2_num","bnkrptcy_num","trds_opnd_past_3_mos_num","inqurs_in_lst_6_mos_num","trds_max_delnqncy_3_in_24_mos","tot_crdt_lmt_amt","satsfctry_bnk_nstlmt_bnkrptcy","avg_cur_bal_all_trds_amt","trds_max_delnqncy_4_in_24_mos","trds_mop_equal_4_num","avg_bal_all_fin_trds_amt","tot_fin_hi_crdt_lmt_amt","trds_opnd_past_6_mos_num","crrntly_actv_nstlmt_trds_num","trds_wth_histmop_3_num","dergtry_publc_rcds_num","trds_mop_equal_3_num","trds_mop_equal_2_num","avg_bal_all_bankcard_trds_amt","trds_wth_histmop_5_num","trds_opnd_past_12_mos_num","crrntly_actv_retl_trds_num","trds_wth_histmop_4_num","avg_cur_bal_mortgg_trds_amt","trds_mop_equal_5_num","trds_wth_histmop_5n24_mos_num","tot_mortgg_hi_crdts_lmt_amt","crrntly_actv_dept_stre_trdsnum","nstlmt_trds_num","dept_stre_trds_bal_gt_0_num","crrntly_actv_fin_trds_num","trds_opnd_past_24_mos_num","auto_trds_verfd_in_6_mos","fin_trds_num","trds_cur_past_due_num","psnl_fin_trds_num","mos_snc_rcnt_fin_trds_updt_num","auto_trds_num","opn_bnk_rev_trd_opnd_12mos_num","cableover_sub_count","dept_stre_trds_lmt_gt_75_pcntg","trds_curmop_5_or_mre_24_mo","trds_curmop_4_or_mre_24_mo_num","trds_curmop_3_or_mre_24_mo_num","uversedata_sub_count","trds_mop_equal_4_or_gt_num","dept_stre_trds_verfy_12_mo_num","uverse_sub_count","trds_mop_equal_3_or_gt_num","dept_stre_trds_lmt_gt_50_pcntg","mos_snc_mst_rcnt_act_num","avg_cur_bal_all_dept_stre_amt","fibertv_sub_count","fin_trds_lmt_gt_75_pcntg","tot_dept_stre_hi_crdt_lmt_amt","trds_cur_bal_amt_gt_0_num","mos_snc_rcnt_fin_trds_opnd_num","clctn_in_24_mo_num","rcnt_trd_opn_mos_num","trds_num","bnk_revlvng_trds_bal_gt_0_num","bnk_revlvng_inqurs_6_mo","bnk_revlvng_trds_lmt_gt_50_pcn","rtio_tot_bal_hi_crdt_dept_stre","satsfctry_bankcard_trds_num","satsfctry_bnk_revlvng_trds_num","opn_bankcard_trds_num","satsfctry_trds_num","mos_snc_rcnt_dept_trd_opnd_num","satsfctry_trds_opn_3_mos_num","trds_gt_75_lmt_pcntg","rtio_tot_bal_hi_crdt_bnk_trds","rtio_tot_cur_bal_hi_trds_num","trds_never_dlqnt_pcntg","oldst_trd_opn_mos_num","dish_sub_count","cable_sub_count","cablemodem_sub_count","cableprim_sub_count","directv_sub_count","dsl_sub_count","dbs_sub_count","hdtv_sub_count","hh_count","paytv_sub_count","snc_rcnt_bnk_revlvng_trd_opnd","fiosdata_sub_count","fiberdata_sub_count","comcast_cable_sub_count","charter_cable_sub_count")

train <- train[,!(names(train) %in% Drop)]




###############################
# score model optimal binning #
###############################

backup <- train
#train <- backup
.libPaths('/apps/EnterpriseAnalytics/R_Projects/sharedLibrary')
library(smbinning) 
# Y-Binary response variable(0,1), X-continuous variable(predictor), P- percentage of records per bin.


#tenure_binning <-smbinning(df=train,y="is_restart",x="tenure",p=0.05)
#tenure_binning
#train$tenure_cat <- cut(as.numeric(train$tenure),breaks = c(-Inf,338,454,536,756,858,925,1263,1485,2059,2938,4134,Inf), labels = c("1","2","3","4","5","6","7","8","9","10","11","12"), right = TRUE)
train$tenure_cat <- cut(as.numeric(train$tenure),breaks = c(-Inf,895,1263,1480,1741,3293,Inf), labels = c("1","2","3","4","5","6"), right = TRUE)


#credit_score_binning <-smbinning(df=train,y="is_restart",x="credit_score",p=0.05)
#credit_score_binning
#train$credit_score_cat <- cut(as.numeric(train$credit_score),breaks = c(-Inf,0,555,575,595,615,635,655,675,685,705,Inf), labels = c("1","2","3","4","5","6","7","8","9","10","11"), right = TRUE)
train$credit_score_cat <- cut(as.numeric(train$credit_score),breaks = c(-Inf,205,535,565,575,595,615,635,655,675,695,705,Inf), labels = c("1","2","3","4","5","6","7","8","9","10","11","12"), right = TRUE)

#ra_age_binning <-smbinning(df=train,y="is_restart",x="ra_age",p=0.05)
#ra_age_binning 
#train$ra_age_cat <- cut(as.numeric(train$ra_age),breaks = c(-Inf,3,5,Inf), labels = c("1","2","3"), right = TRUE)
train$ra_age_cat <- cut(as.numeric(train$ra_age),breaks = c(-Inf,3,5,Inf), labels = c("1","2","3"), right = TRUE)


#no_of_receivers_binning <-smbinning(df=train,y="is_restart",x="no_of_receivers",p=0.05)
#no_of_receivers_binning 
#train$no_of_receivers_cat <- cut(as.numeric(train$no_of_receivers),breaks = c(-Inf,1,2,3,4,Inf), labels = c("1","2","3","4","5"), right = TRUE)
train$no_of_receivers_cat <- cut(as.numeric(train$no_of_receivers),breaks = c(-Inf,1,2,Inf), labels = c("1","2","3"), right = TRUE)


#aggr_amt_derog_pub_rcnt_36_mos_binning <-smbinning(df=train,y="is_restart",x="aggr_amt_derog_pub_rcnt_36_mos",p=0.05)
#aggr_amt_derog_pub_rcnt_36_mos_binning
train$aggr_amt_derog_pub_rcnt_36_mos_cat <- cut(as.numeric(train$aggr_amt_derog_pub_rcnt_36_mos),breaks = c(-Inf,16156, 22041, 28155, 35838,Inf), labels = c("1","2","3","4","5"), right = TRUE)


#tot_bnk_revlvng_hi_crdt_amt_binning <-smbinning(df=train, y="is_restart", x="tot_bnk_revlvng_hi_crdt_amt",p=0.05)
#tot_bnk_revlvng_hi_crdt_amt_binning
train$tot_bnk_revlvng_hi_crdt_amt_cat <- cut(as.numeric(train$tot_bnk_revlvng_hi_crdt_amt),breaks = c(-Inf,272.4101,1248.2800,3245.1200,5132.3500,7294.7100,  9955.1201,12792.7200,14845.4500,20222.9700,24100.3800,37125.0300,Inf), labels = c("1","2","3","4","5","6","7","8","9","10","11","12"), right = TRUE)


#satsfctry_dept_stre_trds_num_binning <-smbinning(df=train, y="is_restart", x="satsfctry_dept_stre_trds_num",p=0.05)
#satsfctry_dept_stre_trds_num_binning
train$satsfctry_dept_stre_trds_num_cat <- cut(as.numeric(train$satsfctry_dept_stre_trds_num),breaks = c(-Inf,0.0701,0.1800,0.3100,0.3700,0.4900,0.6000,0.7100,0.8000,0.9400,Inf), labels = c("1","2","3","4","5","6","7","8","9","10"), right = TRUE)


#trds_mop_equal_2_or_gt_num_binning <-smbinning(df=train, y="is_restart", x="trds_mop_equal_2_or_gt_num",p=0.05)
#trds_mop_equal_2_or_gt_num_binning
#train$trds_mop_equal_2_or_gt_num_cat <- cut(as.numeric(train$trds_mop_equal_2_or_gt_num),breaks = c(-Inf,0.1800,0.3200,0.4200,0.4800,0.6200,0.7400,0.8101,1.0900,1.7100,Inf), labels = c("1","2","3","4","5","6","7","8","9","10"), right = TRUE)
train$trds_mop_equal_2_or_gt_num_cat <- cut(as.numeric(train$trds_mop_equal_2_or_gt_num),breaks = c(-Inf,0.1900,0.2900,0.4300,0.6200,0.8101,1.0900,1.7000,Inf), labels = c("1","2","3","4","5","6","7","8"), right = TRUE)

#snc_rcnt_delnqncy_mos_binning <-smbinning(df=train, y="is_restart", x="snc_rcnt_delnqncy_mos",p=0.05)
#snc_rcnt_delnqncy_mos_binning
train$snc_rcnt_delnqncy_mos_cat <- cut(as.numeric(train$snc_rcnt_delnqncy_mos),breaks = c(-Inf,7.00,23.36,26.65,31.45,36.90,Inf), labels = c("1","2","3","4","5","6"), right = TRUE)


#dept_stre_trds_num_binning <-	smbinning(df=train, y="is_restart", x="dept_stre_trds_num",p=0.05)
#dept_stre_trds_num_binning
train$dept_stre_trds_num_cat <- cut(as.numeric(train$dept_stre_trds_num),breaks = c(-Inf,0.3700,0.6300,0.9900,1.1700,1.3701,1.5500,1.7600,2.0500,Inf), labels = c("1","2","3","4","5","6","7","8","9"), right = TRUE)


#highst_delnqncy_ever_on_trdnum_binning <-smbinning(df=train,y="is_restart",x="highst_delnqncy_ever_on_trdnum",p=0.05)
#highst_delnqncy_ever_on_trdnum_binning
train$highst_delnqncy_ever_on_trdnum_cat <- cut(as.numeric(train$highst_delnqncy_ever_on_trdnum),breaks = c(-Inf,4.70,5.37,5.62,6.00,6.50,7.00,Inf), labels = c("1","2","3","4","5","6","7"), right = TRUE)


#fin_trds_lmt_gt_50_pcntg_binning <-smbinning(df=train, y="is_restart", x="fin_trds_lmt_gt_50_pcntg",p=0.05)
#fin_trds_lmt_gt_50_pcntg_binning
train$fin_trds_lmt_gt_50_pcntg_cat <- cut(as.numeric(train$fin_trds_lmt_gt_50_pcntg),breaks = c(-Inf,29.97,49.99,52.15,59.93,63.45,67.60,72.31,89.00,Inf), labels = c("1","2","3","4","5","6","7","8","9"), right = TRUE)


#tot_amt_now_past_due_amt_binning <-	smbinning(df=train, y="is_restart", x="tot_amt_now_past_due_amt",p=0.05)
#tot_amt_now_past_due_amt_binning
#train$tot_amt_now_past_due_amt_cat <- cut(as.numeric(train$tot_amt_now_past_due_amt),breaks = c(-Inf,0.24,6.22,Inf), labels = c("1","2","3"), right = TRUE)
train$tot_amt_now_past_due_amt_cat <- cut(as.numeric(train$tot_amt_now_past_due_amt),breaks = c(-Inf,5.77,24.50,742.25,Inf), labels = c("1","2","3","4"), right = TRUE)


#crrntly_actv_mortgg_trds_num_binning <-	smbinning(df=train, y="is_restart", x="crrntly_actv_mortgg_trds_num",p=0.05)
#crrntly_actv_mortgg_trds_num_binning
train$crrntly_actv_mortgg_trds_num_cat <- cut(as.numeric(train$crrntly_actv_mortgg_trds_num),breaks = c(-Inf,0.0300,0.1000,0.1600,0.2000,0.2300,0.2600,0.3000,0.3401,0.4200,0.5601,Inf), labels = c("1","2","3","4","5","6","7","8","9","10","11"), right = TRUE)


#mst_rcnt_inqry_age_num_binning <-	smbinning(df=train, y="is_restart", x="mst_rcnt_inqry_age_num",p=0.05)
#mst_rcnt_inqry_age_num_binning
#train$mst_rcnt_inqry_age_num_cat <- cut(as.numeric(train$mst_rcnt_inqry_age_num),breaks = c(-Inf,3038,8991,13759,27022,30334,33431,Inf), labels = c("1","2","3","4","5","6","7"), right = TRUE)


#psnl_fin_inqurs_in_24_mos_num_binning <-smbinning(df=train, y="is_restart", x="psnl_fin_inqurs_in_24_mos_num",p=0.05)
#psnl_fin_inqurs_in_24_mos_num_binning
train$psnl_fin_inqurs_in_24_mos_num_cat <- cut(as.numeric(train$psnl_fin_inqurs_in_24_mos_num),breaks = c(-Inf,1.91,4.72,7.16,10.00,11.59,16.90,33.50,Inf), labels = c("1","2","3","4","5","6","7","8"), right = TRUE)


#bnk_crd_trds_gt_75_lmt_pcntg_binning <-	smbinning(df=train, y="is_restart", x="bnk_crd_trds_gt_75_lmt_pcntg",p=0.05)
#bnk_crd_trds_gt_75_lmt_pcntg_binning
train$bnk_crd_trds_gt_75_lmt_pcntg_cat <- cut(as.numeric(train$bnk_crd_trds_gt_75_lmt_pcntg),breaks = c(-Inf,6.2500,18.7300,24.9100,33.1400,49.7701,65.8800,Inf), labels = c("1","2","3","4","5","6","7"), right = TRUE)


#mos_snc_oldst_mortgg_opnd_num_binning <-smbinning(df=train, y="is_restart", x="mos_snc_oldst_mortgg_opnd_num",p=0.05)
#mos_snc_oldst_mortgg_opnd_num_binning
#train$mos_snc_oldst_mortgg_opnd_num_cat <- cut(as.numeric(train$mos_snc_oldst_mortgg_opnd_num),breaks = c(-Inf,19.50,147.39,154.72,183.46,Inf), labels = c("1","2","3","4","5"), right = TRUE)
train$mos_snc_oldst_mortgg_opnd_num_cat <- cut(as.numeric(train$mos_snc_oldst_mortgg_opnd_num),breaks = c(-Inf,19.50,147.90,183.57,Inf), labels = c("1","2","3","4"), right = TRUE)

#snc_rcnt_bankcard_trd_opn_mos_binning <-smbinning(df=train, y="is_restart", x="snc_rcnt_bankcard_trd_opn_mos",p=0.05)
#snc_rcnt_bankcard_trd_opn_mos_binning
train$snc_rcnt_bankcard_trd_opn_mos_cat <- cut(as.numeric(train$snc_rcnt_bankcard_trd_opn_mos),breaks = c(-Inf,51.88,64.77,Inf), labels = c("1","2","3"), right = TRUE)


#opn_trds_num_binning <-	smbinning(df=train, y="is_restart", x="opn_trds_num",p=0.05)
#opn_trds_num_binning
train$opn_trds_num_cat <- cut(as.numeric(train$opn_trds_num),breaks = c(-Inf,1.4000,2.4600,2.8900,3.4100,3.6600,3.9900,4.2700,4.6901,5.1400,5.8100,6.5200,Inf), labels = c("1","2","3","4","5","6","7","8","9","10","11","12"), right = TRUE)


#satsfctry_trds_opn_24_mos_num_binning <-smbinning(df=train, y="is_restart", x="satsfctry_trds_opn_24_mos_num",p=0.05)
#satsfctry_trds_opn_24_mos_num_binning
train$satsfctry_trds_opn_24_mos_num_cat <- cut(as.numeric(train$satsfctry_trds_opn_24_mos_num),breaks = c(-Inf,0.00,0.52,0.96,1.54,2.18,2.41,2.67,3.08,3.49, 4.02,4.78,11.91,Inf), labels = c("1","2","3","4","5","6","7","8","9","10","11","12","13"), right = TRUE)


#bnk_revlvng_trds_num_binning <-	smbinning(df=train, y="is_restart", x="bnk_revlvng_trds_num",p=0.05)
#bnk_revlvng_trds_num_binning
train$bnk_revlvng_trds_num_cat <- cut(as.numeric(train$bnk_revlvng_trds_num),breaks = c(-Inf,0.4800,1.0700,1.3601,1.6600,2.2100,2.7300,3.1100,3.3700,3.8400,4.2800,4.7000,5.7700,Inf), labels = c("1","2","3","4","5","6","7","8","9","10","11","12","13"), right = TRUE)


#trds_gt_50_lmt_pcntg_binning <-	smbinning(df=train, y="is_restart", x="trds_gt_50_lmt_pcntg",p=0.05)
#trds_gt_50_lmt_pcntg_binning
train$trds_gt_50_lmt_pcntg_cat <- cut(as.numeric(train$trds_gt_50_lmt_pcntg),breaks = c(-Inf,37.3300,40.3800,43.6500,48.8900,58.9800,69.0401,78.2600,Inf), labels = c("1","2","3","4","5","6","7","8"), right = TRUE)


#snc_oldst_retl_trd_opn_mos_num_binning <-smbinning(df=train,y="is_restart",x="snc_oldst_retl_trd_opn_mos_num",p=0.05)
#snc_oldst_retl_trd_opn_mos_num_binning
train$snc_oldst_retl_trd_opn_mos_num_cat <- cut(as.numeric(train$snc_oldst_retl_trd_opn_mos_num),breaks = c(-Inf,62.71,83.06,97.81,147.32,172.17,194.80,Inf), labels = c("1","2","3","4","5","6","7"), right = TRUE)


#mos_on_file_num_binning <-	smbinning(df=train, y="is_restart", x="mos_on_file_num",p=0.05)
#mos_on_file_num_binning
train$mos_on_file_num_cat <- cut(as.numeric(train$mos_on_file_num),breaks = c(-Inf,139.12,187.20,232.55,250.70,268.26,285.19,309.68,Inf), labels = c("1","2","3","4","5","6","7","8"), right = TRUE)


#snc_oldst_bankcard_trd_opn_mos_binning <-smbinning(df=train,y="is_restart",x="snc_oldst_bankcard_trd_opn_mos",p=0.05)
#snc_oldst_bankcard_trd_opn_mos_binning
train$snc_oldst_bankcard_trd_opn_mos_cat <- cut(as.numeric(train$snc_oldst_bankcard_trd_opn_mos),breaks = c(-Inf,88.71,107.09,121.71,156.38,182.27,198.40,221.93,249.17,Inf), labels = c("1","2","3","4","5","6","7","8","9"), right = TRUE)


#broadband_sub_count_binning <-	smbinning(df=train, y="is_restart", x="broadband_sub_count",p=0.05)
#broadband_sub_count_binning
train$broadband_sub_count_cat <- cut(as.numeric(train$broadband_sub_count),breaks = c(-Inf,208.14,233.37,256.54,396.84,689.30,Inf), labels = c("1","2","3","4","5","6"), right = TRUE)


#tv_hh_count_binning <-	smbinning(df=train, y="is_restart", x="tv_hh_count",p=0.05)
#tv_hh_count_binning
train$tv_hh_count_cat <- cut(as.numeric(train$tv_hh_count),breaks = c(-Inf,1403,10450,15991,Inf), labels = c("1","2","3","4"), right = TRUE)



Drop <- c("aggr_amt_derog_pub_rcnt_36_mos","tot_bnk_revlvng_hi_crdt_amt","satsfctry_dept_stre_trds_num","trds_mop_equal_2_or_gt_num","snc_rcnt_delnqncy_mos","dept_stre_trds_num","highst_delnqncy_ever_on_trdnum","fin_trds_lmt_gt_50_pcntg","tot_amt_now_past_due_amt","ra_age","crrntly_actv_mortgg_trds_num","mst_rcnt_inqry_age_num","psnl_fin_inqurs_in_24_mos_num","bnk_crd_trds_gt_75_lmt_pcntg","mos_snc_oldst_mortgg_opnd_num","snc_rcnt_bankcard_trd_opn_mos","opn_trds_num","satsfctry_trds_opn_24_mos_num","snc_rcnt_bnk_revlvng_trd_opnd","bnk_revlvng_trds_num","trds_gt_50_lmt_pcntg","snc_oldst_retl_trd_opn_mos_num","mos_on_file_num","snc_oldst_bankcard_trd_opn_mos","broadband_sub_count","tv_hh_count","tenure","tenure_days","credit_score","no_of_receivers","loyalty_call_count","ca_number","mst_rcnt_inqry_age_num_cat")

train <- train[,!(names(train) %in% Drop)]



#######################################

backup1 <- train 

var = names(train)
Profile.Table = data.frame(NULL)
attach(train)
for(i in 3:ncol(train)){
   
   if (is.factor(train[,i])){


   table1<- table(train[,i],train[,1])
   table2<-as.data.frame(prop.table(table1,2))  
   colnames(table2) = c("Category",var[1],"ColPct")
   
   table1 = as.data.frame(table1) 
   colnames(table1) = c("Category",var[1],"Freq")

   table20 = table2[table2$is_restart==0,]
   colnames(table20)=c("Category",var[1],"Roll_PCT")
   
   table10 = table1[table1$is_restart==0,] 
   colnames(table10)=c("Category",var[1],"Roll_FREQ")

   table120= merge(table10,table20,by=c("Category",var[1]))

   table21 = table2[table2$is_restart==1,]
   colnames(table21)=c("Category",var[1],"Resolved_PCT")
   
   table11 = table1[table1$is_restart==1,] 
   colnames(table11)=c("Category",var[1],"Resolved_FREQ")

   table121= merge(table11,table21,by=c("Category",var[1]))

   table1201 = merge(table121[,-2],table120[,-2],by="Category")

   attach(table1201)
     for(j in 1:nrow(table1201)){
         if(Roll_PCT[j] > 0 & Resolved_PCT[j]>0)
            { 
             table1201$WOE[j] =log(Resolved_PCT[j]/Roll_PCT[j],base=exp(1))
            } 
         else
            {
             table1201$WOE[j] = 0 
            } 
         ##Variable[j]=c(var[i])
                                 }
   detach(table1201)
   Variable=rep(factor(var[i]),times=nrow(table1201))
   table1201 = cbind(Variable,table1201)
   table1201 = table1201[order(-table1201$WOE),]
   Profile.Table = rbind(Profile.Table,table1201)
                          }
                       }

detach(train)

write.csv(Profile.Table,file ="/users/mis/mohafnu/restarts/Profile.Table.csv")
#####################################
##  Cardinality Reduction - Macro  ##
#####################################
#####################################
                            

Variable = "county"

Information.Matrix = data.frame(NULL)
for(maxc in seq(6,32,by=1))
{

Profile.Table.1 = Profile.Table[Profile.Table$Variable==Variable & Profile.Table$Roll_FREQ > 0,]
set.seed(1)
fit= kmeans(Profile.Table.1[,7],maxc)
Profile.Table.1 = cbind(Profile.Table.1,Cluster=fit$cluster)
Profile.Table.0 = Profile.Table[Profile.Table$Variable==Variable & Profile.Table$Roll_FREQ == 0,]

for(i in 1:nrow(Profile.Table.0))
{
Profile.Table.0$Cluster[i] = Profile.Table.1[1,8]
}

Profile.Final = rbind(Profile.Table.1,Profile.Table.0)

Profile.Summary.0 = as.data.frame(aggregate(Profile.Final$Roll_PCT,by=list(Profile.Final$Cluster),FUN=sum))
colnames(Profile.Summary.0) = c("Cluster","Roll_Pct")

Profile.Summary.1 = as.data.frame(aggregate(Profile.Final$Resolved_PCT,by=list(Profile.Final$Cluster),FUN=sum))
colnames(Profile.Summary.1) = c("Cluster","Resolved_Pct")

Profile.Summary = merge(Profile.Summary.0,Profile.Summary.1,by="Cluster")

attach(Profile.Summary)
Profile.Summary$Info.Val = (Resolved_Pct-Roll_Pct)*log(Resolved_Pct/Roll_Pct,base=exp(1))
Profile.Summary$WOE = log(Resolved_Pct/Roll_Pct,base=exp(1))
Profile.Summary = Profile.Summary[order(-Profile.Summary$WOE),]
Profile.Summary$Regroup = index(Profile.Summary$WOE)
detach(Profile.Summary)

Info.Val = sum(Profile.Summary$Info.Val)

Info.Tbl = as.data.frame(cbind(maxc,Info.Val))

Information.Matrix = rbind(Information.Matrix,Info.Tbl)              

}

print(Information.Matrix)

plot(Information.Matrix$maxc,Information.Matrix$Info.Val,type="l",col="blue",lwd=2.5)

Profile.Table.1 = Profile.Table[Profile.Table$Variable==Variable & Profile.Table$Roll_FREQ > 0,]
set.seed(1)
fit= kmeans(Profile.Table.1[,7],28)
Profile.Table.1 = cbind(Profile.Table.1,Cluster=fit$cluster)
Profile.Table.0 = Profile.Table[Profile.Table$Variable==Variable & Profile.Table$Roll_FREQ == 0,]

for(i in 1:nrow(Profile.Table.0))
{
Profile.Table.0$Cluster[i] = Profile.Table.1[1,8]
}

Profile.Final = rbind(Profile.Table.1,Profile.Table.0)

Profile.Summary.0 = as.data.frame(aggregate(Profile.Final$Roll_PCT,by=list(Profile.Final$Cluster),FUN=sum))
colnames(Profile.Summary.0) = c("Cluster","Roll_Pct")

Profile.Summary.1 = as.data.frame(aggregate(Profile.Final$Resolved_PCT,by=list(Profile.Final$Cluster),FUN=sum))
colnames(Profile.Summary.1) = c("Cluster","Resolved_Pct")

Profile.Summary = merge(Profile.Summary.0,Profile.Summary.1,by="Cluster")

attach(Profile.Summary)
Profile.Summary$Info.Val = (Resolved_Pct-Roll_Pct)*log(Resolved_Pct/Roll_Pct,base=exp(1))
Profile.Summary$WOE = log(Resolved_Pct/Roll_Pct,base=exp(1))
Profile.Summary = Profile.Summary[order(-Profile.Summary$WOE),]
Profile.Summary$Regroup = index(Profile.Summary$WOE)
detach(Profile.Summary)


Profile.Final = merge(Profile.Final,Profile.Summary[,-c(2:5)],by="Cluster")

subset = c("Category","Regroup")
Profile.Select = Profile.Final[,names(Profile.Final) %in% subset]
colnames(Profile.Select)  = c("county","county_cat")

train = merge(train,Profile.Select, by="county")
train$county_cat <- as.factor(train$county_cat)

##
write.csv(train,file ="/users/mis/mohafnu/restarts/train.cluster.csv")
Drop <- c("county","primary_cable_sp_name","primary_telco_sp_name","prime_post_office_name")
train <- train[,!(names(train) %in% Drop)]
######################## Data Readfine #####################
backup2 <- train
train <- backup2
#################### WOE mapping
#Resolved = 1
#Roll = 0
attach(train)
strt <- 3
end <- ncol(train)
var <- names(train)
attach(train)

for(i in 3 :ncol(train))      
{    
   if (is.factor(train[,3]) & length(unique(train[,3])) > 1)  {
      
      table1 <- table(train[,3],train[,1])
      table2 <- as.data.frame(prop.table(table1,2))
      colnames(table2) <- c(var[i],var[1],"Freq")

      table20 <- table2[table2$is_restart==0,]
      colnames(table20) <- c(var[i],var[1],"Roll")

      table21 <- table2[table2$is_restart==1,]
      colnames(table21) <- c(var[i],var[1],"Resolved")

      table201 <- merge(table21[,-2],table20[,-2],by=var[i])

      attach(table201)
        for(j in 1:nrow(table201))
		
	   {
           if(Roll[j] > 0 & Resolved[j] > 0)
            { 
             table201$WOE[j] <- round(log(Resolved[j]/Roll[j]),digits=4) ### Weight of Evidence
            } 
            else
            {
             table201$WOE[j] = 0 
            } 
       }
      detach(table201)
      save(table201,file=paste(var[i],"_W",".rda",sep=""))
      train <- merge(train,table201[,-(2:3)],by=var[i],sort=TRUE)
      new <- paste(var[i],"w",sep="_")
      names(train)[names(train)=="WOE"]=new
      train <- train[,-1]
	} 
}


write.csv(train,file ="/users/mis/mohafnu/restarts/train_WOE.csv")




###############################################################   Step: 7
#backup3 <- train
#train <- backup3

set.seed(12345)
#data partition(train, test)

Drop <- c("payment_w","bnk_crd_trds_gt_75_lmt_pcntg_cat_w","standard_specialty_cc_w","equip_charge_amt_w","snc_oldst_retl_trd_opn_mos_num_cat_w","upscale_retail_cc_w","mutual_funds_w","broadband_sub_count_cat_w","snc_rcnt_bankcard_trd_opn_mos_cat_w","psnl_fin_inqurs_in_24_mos_num_cat_w","langassim_w","dept_stre_trds_num_cat_w","crrntly_actv_mortgg_trds_num_cat_w","boating_sailing_w","adults_65_74_w","stock_or_bonds_w","highst_delnqncy_ever_on_trdnum_cat_w","bbhome_ever_w","fin_trds_lmt_gt_50_pcntg_cat_w","snc_rcnt_delnqncy_mos_cat_w","encore_ever_w","adults_55_64_w","satsfctry_trds_opn_24_mos_num_cat_w","golf_w","travel_in_the_usa_w","bnk_revlvng_trds_num_cat_w","mos_on_file_num_cat_w","tv_hh_count_cat_w","investments_w","vip_wo_dvr_w","travel_entertainment_cc_w","vip_w_dvr_w","satsfctry_dept_stre_trds_num_cat_w","trds_gt_50_lmt_pcntg_cat_w","legacy_unit_w","opn_trds_num_cat_w","snc_oldst_bankcard_trd_opn_mos_cat_w","tot_bnk_revlvng_hi_crdt_amt_cat_w","rv_vacations_w","hhold_age_w","aggr_amt_derog_pub_rcnt_36_mos_cat_w","numb_child_w","ra_age_cat_w","commitment_w","sling_w","payment_method_w")
train <- train[,!(names(train) %in% Drop)]


train$is_restart <- as.factor(train$is_restart)
Split<-createDataPartition(train[,1],p=0.60,list=FALSE,times=1)
trainModel<-train[Split,]
testModel<-train[-Split,]
trainModel$acct_no <- NULL


logistic_model <- glm(is_restart ~ ., data=trainModel, family=binomial(link="logit"))
summary(logistic_model)
varImp(logistic_model)
VIF_values <- as.data.frame(vif(logistic_model))
VIF_values


#### random forest new

#parallel processing
registerDoMC(cores = 16)  #these many parallel executions will hap#pen
ctrl <- trainControl(method = "repeatedcv",number = 10, repeats = 10)  #method for training cross-validation #and 10-folds will be created
#
grid_rf <- expand.grid(.mtry = c(2, 4, 8, 16)) # model selection parameter in this case mtry.
# Both trainControl and expand.grid is provided by caret
system.time(model_randomForest <- train(is_restart~ ., data = trainModel,
											method = "rf", metric = "Kappa", 
											trControl = ctrl,ntree=1000,
											tuneGrid = grid_rf)) #metric is used to measure the model accuracy.

testModel$predRFprob_class <- predict(model_randomForest , testModel, type = "raw")
conftreebag_prob <- confusionMatrix(testModel$predRFprob_class,testModel[,1], positive = levels(testModel$is_restart)[2])
conftreebag_prob

saveRDS(model_randomForest, "/users/mis/mohafnu/restarts/RandomForest.RDS")

Confusion Matrix and Statistics

Confusion Matrix and Statistics

          Reference
Prediction     0     1
         0  7492  1395
         1  1847 10650
                                               
               Accuracy : 0.8484               
                 95% CI : (0.8435, 0.8532)     
    No Information Rate : 0.5633               
    P-Value [Acc > NIR] : < 0.00000000000000022
                                               
                  Kappa : 0.6902               
 Mcnemars Test P-Value : 0.00000000000000236  
                                               
            Sensitivity : 0.8842               
            Specificity : 0.8022               
         Pos Pred Value : 0.8522               
         Neg Pred Value : 0.8430               
             Prevalence : 0.5633               
         Detection Rate : 0.4980               
   Detection Prevalence : 0.5844               
      Balanced Accuracy : 0.8432               
                                               
       'Positive' Class : 1                    

                                    Overall
disco_type_w                        100.000
count_of_latecharge_w                87.150
all_star_w                           79.169
core_programming_w                   53.877
county_cat_w                         41.579
credit_score_cat_w                   35.530
core_international_w                 31.470
trds_mop_equal_2_or_gt_num_cat_w     25.630
tenure_cat_w                         17.480
payment_method_w                     15.648
model_no_w                           14.386
tot_amt_now_past_due_amt_cat_w       12.610
mos_snc_oldst_mortgg_opnd_num_cat_w  12.279
no_of_receivers_cat_w                12.100
line_of_business_w                    8.460
ra_age_cat_w                          7.927
commitment_w                          4.856
sling_w                               0.000


##
system.time(Modeltreebag <- train(is_restart ~., data = trainModel, method="treebag", metric = "Kappa", trControl = trainControl(method = "cv", number = 16, allowParallel=TRUE)))
testModel$predtreebag_class <- predict(Modeltreebag, testModel, type = "raw")
conftreebag_prob <- confusionMatrix(testModel$predtreebag_class,testModel[,1], positive = levels(testModel$is_restart)[2])
conftreebag_prob
saveRDS(Modeltreebag, "/users/mis/mohafnu/restarts/Modeltreebag.RDS")
Confusion Matrix and Statistics
          Reference
Prediction     0     1
         0  7472  1675
         1  1867 10370
                                               
               Accuracy : 0.8344               
                 95% CI : (0.8293, 0.8393)     
    No Information Rate : 0.5633               
    P-Value [Acc > NIR] : < 0.00000000000000022
                                               
                  Kappa : 0.6626               
 Mcnemars Test P-Value : 0.001331             
                                               
            Sensitivity : 0.8609               
            Specificity : 0.8001               
         Pos Pred Value : 0.8474               
         Neg Pred Value : 0.8169               
             Prevalence : 0.5633               
         Detection Rate : 0.4849               
   Detection Prevalence : 0.5723               
      Balanced Accuracy : 0.8305               
                                               
       'Positive' Class : 1                    
                                               

treebag variable importance

                                    Overall
all_star_w                          100.000
count_of_latecharge_w                90.170
disco_type_w                         83.095
credit_score_cat_w                   62.135
core_programming_w                   61.234
county_cat_w                         52.492
trds_mop_equal_2_or_gt_num_cat_w     37.788
payment_method_w                     34.780
core_international_w                 27.104
tenure_cat_w                         24.157
model_no_w                           22.710
no_of_receivers_cat_w                20.624
mos_snc_oldst_mortgg_opnd_num_cat_w  19.732
tot_amt_now_past_due_amt_cat_w       19.636
line_of_business_w                   14.571
ra_age_cat_w                         14.071
commitment_w                          6.002
sling_w                               0.000

system.time(Modeladabag <- train(is_restart ~., data = trainModel, method="ada", metric = "Kappa", trControl = trainControl(method = "cv", number = 16, allowParallel=TRUE)))
#predict class
testModel$predadabag_class <- predict(Modeladabag, testModel)
conftreebag_class <- confusionMatrix(testModel$predadabag_class, testModel[,1], positive = levels(testModel$is_restart)[2])
conftreebag_class
varImp(Modeladabag)
saveRDS(Modeladabag, "/users/mis/mohafnu/restarts/Modeladabag.RDS")
Confusion Matrix and Statistics

          Reference
Prediction     0     1
         0  7337  1420
         1  2002 10625
                                               
               Accuracy : 0.84                 
                 95% CI : (0.835, 0.8449)      
    No Information Rate : 0.5633               
    P-Value [Acc > NIR] : < 0.00000000000000022
                                               
                  Kappa : 0.6724               
 Mcnemars Test P-Value : < 0.00000000000000022
                                               
            Sensitivity : 0.8821               
            Specificity : 0.7856               
         Pos Pred Value : 0.8415               
         Neg Pred Value : 0.8378               
             Prevalence : 0.5633               
         Detection Rate : 0.4969               
   Detection Prevalence : 0.5905               
      Balanced Accuracy : 0.8339               
                                               
       'Positive' Class : 1                    

	   
                                    Importance
all_star_w                             100.000
disco_type_w                            99.012
count_of_latecharge_w                   82.240
credit_score_cat_w                      66.447
county_cat_w                            46.330
payment_method_w                        45.170
core_programming_w                      43.562
tenure_cat_w                            33.953
core_international_w                    26.218
trds_mop_equal_2_or_gt_num_cat_w        22.475
commitment_w                            20.163
model_no_w                              18.260
tot_amt_now_past_due_amt_cat_w          13.671
mos_snc_oldst_mortgg_opnd_num_cat_w     10.489
line_of_business_w                       7.913
ra_age_cat_w                             7.002
no_of_receivers_cat_w                    6.089
sling_w                                  0.000

predAdabag_class <- predict.boosting(Modeladabag,testModel, type='class')
names(predAdabag_class)
testModel$predAdabag_Fact_class <- predAdabag_class$class 

ModelAdabag_class <- confusionMatrix(testModel$predAdabag_Fact_class, testModel[,1])
ModelAdabag_class
######
library('ROCR')

pred = prediction(testModel$probRF[1:nrow(testModel),2], testModel$is_restart[1:nrow(testModel)])
roc = performance(pred, "tpr", "fpr")

plot(roc, lwd=2, colorize=TRUE)
lines(x=c(0, 1), y=c(0, 1), col="black", lwd=1)

auc = performance(pred, "auc")
auc = unlist(auc@y.values)
auc

library(ROCR)
pred <- prediction( testModel$probRF_churn[1:nrow(testModel)], testModel$is_restart[1:nrow(testModel)])
perf <- performance(pred,"tpr","fpr")
plot(perf)
## precision/recall curve (x-axis: recall, y-axis: precision)
perf1 <- performance(pred, "prec", "rec")
plot(perf1)
## sensitivity/specificity curve (x-axis: specificity,
## y-axis: sensitivity)
perf1 <- performance(pred, "sens", "spec")
plot(perf1)

########################### 
