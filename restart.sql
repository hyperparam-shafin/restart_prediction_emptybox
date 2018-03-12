create external table sbx_enterpriseanalytics.empty_box_train_010114_123116(
ACCT_NO string, 
CA_NUMBER string, 
LAST_UPDATE_DATE string , 
DISCO_TYPE string, 
EQUIP_STATUS string, 
EQUIP_CHARGE_AMT double, 
MODEL_NO string, 
RA_AGE string,
AVG_BILL double,
COUNT_OF_LATE_CHARGE int)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
location '/user/enterpriseanalytics/sbx_enterpriseanalytics.db/empty_box_train_010114_123116';

create external table sbx_enterpriseanalytics.empty_box_train_010114_123116_bill(
ACCT_NO string,
bill1 double,
bill2 double,
bill3 double,
bill4 double,
bill5 double)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
location '/user/enterpriseanalytics/sbx_enterpriseanalytics.db/empty_box_train_010114_123116_bill';


create external table sbx_enterpriseanalytics.empty_box_curr_bal_disco_date_010114_123116(
ACCT_NO string,
disconnect_date string,
curr_bal string)
ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
location '/user/enterpriseanalytics/sbx_enterpriseanalytics.db/empty_box_curr_bal_disco_date_010114_123116';

create external table sbx_enterpriseanalytics.empty_box_any_restart_180_days_010114_123116(
ACCT_NO string,
dish_restart int)
ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
location '/user/enterpriseanalytics/sbx_enterpriseanalytics.db/empty_box_any_restart_180_days_010114_123116';

create external table sbx_enterpriseanalytics.empty_box_any_restart_train1(
acct_no string,
equip_status string ,
count_of_latecharge int,
core_programming string,
all_star string,
payment_method string,
line_of_business string,
commitment string,
core_international string,
tenure int,
credit_score string,
trds_mop_equal_2_or_gt_num string,
tot_amt_now_past_due_amt string,
bill1 double,
bill2 double,
bill3 double,
bill4 double,
bill5 double,
curr_bal double,
dish_restart_flag int,
no_of_receivers double,
total_devices int,
total_equip_charge_amt double
 )
ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
location '/user/enterpriseanalytics/sbx_enterpriseanalytics.db/empty_box_any_restart_train1';


insert overwrite table sbx_enterpriseanalytics.empty_box_any_restart_train1
select a.acct_no,a.EQUIP_STATUS,a.COUNT_OF_LATE_CHARGE,c.core_programming,c.all_star,c.payment_method,c.line_of_business,c.commitment,c.core_international,c.tenure,c.credit_score,c.trds_mop_equal_2_or_gt_num,c.tot_amt_now_past_due_amt,b.bill1 ,b.bill2 ,b.bill3 ,b.bill4,b.bill5,
d.curr_bal,e.dish_restart,c.no_of_receivers,count( a.MODEL_NO),sum(a.EQUIP_CHARGE_AMT)
from sbx_enterpriseanalytics.empty_box_train_010114_123116 a JOIN cust_aggr.cust_aggr c
ON a.acct_no=c.acct_num
LEFT JOIN sbx_enterpriseanalytics.empty_box_train_010114_123116_bill b
ON a.acct_no=b.ACCT_NO
LEFT JOIN sbx_enterpriseanalytics.empty_box_curr_bal_disco_date_010114_123116 d
ON a.acct_no=d.ACCT_NO
LEFT JOIN sbx_enterpriseanalytics.empty_box_any_restart_180_days_010114_123116 e
ON a.acct_no=e.ACCT_NO
where DISCO_TYPE='NP'
and EQUIP_STATUS='RESTARTED'
group by a.acct_no,a.EQUIP_STATUS,a.COUNT_OF_LATE_CHARGE,c.core_programming,c.all_star,c.payment_method,c.line_of_business,c.commitment,c.core_international,c.tenure,c.credit_score,c.trds_mop_equal_2_or_gt_num,c.tot_amt_now_past_due_amt,b.bill1 ,b.bill2 ,b.bill3 ,b.bill4,b.bill5,
d.curr_bal,e.dish_restart,c.no_of_receivers


create external table sbx_enterpriseanalytics.empty_box_any_restart_train0(
acct_no string,
equip_status string ,
count_of_latecharge int,
core_programming string,
all_star string,
payment_method string,
line_of_business string,
commitment string,
core_international string,
tenure int,
credit_score string,
trds_mop_equal_2_or_gt_num string,
tot_amt_now_past_due_amt string,
bill1 double,
bill2 double,
bill3 double,
bill4 double,
bill5 double,
curr_bal double,
dish_restart_flag int,
no_of_receivers double,
total_devices int,
total_equip_charge_amt double
 )
ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
location '/user/enterpriseanalytics/sbx_enterpriseanalytics.db/empty_box_any_restart_train0';

insert overwrite table sbx_enterpriseanalytics.empty_box_any_restart_train0
select a.acct_no,a.EQUIP_STATUS,a.COUNT_OF_LATE_CHARGE,c.core_programming,c.all_star,c.payment_method,c.line_of_business,c.commitment,c.core_international,c.tenure,c.credit_score,c.trds_mop_equal_2_or_gt_num,c.tot_amt_now_past_due_amt,b.bill1 ,b.bill2 ,b.bill3 ,b.bill4,b.bill5,
d.curr_bal,e.dish_restart,c.no_of_receivers,count( a.MODEL_NO),sum(a.EQUIP_CHARGE_AMT)
from sbx_enterpriseanalytics.empty_box_train_010114_123116 a JOIN cust_aggr.cust_aggr c
ON a.acct_no=c.acct_num
LEFT JOIN sbx_enterpriseanalytics.empty_box_train_010114_123116_bill b
ON a.acct_no=b.ACCT_NO
LEFT JOIN sbx_enterpriseanalytics.empty_box_curr_bal_disco_date_010114_123116 d
ON a.acct_no=d.ACCT_NO
LEFT JOIN sbx_enterpriseanalytics.empty_box_any_restart_180_days_010114_123116 e
ON a.acct_no=e.ACCT_NO
where DISCO_TYPE='NP'
and EQUIP_STATUS!='RESTARTED'
group by a.acct_no,a.EQUIP_STATUS,a.COUNT_OF_LATE_CHARGE,c.core_programming,c.all_star,c.payment_method,c.line_of_business,c.commitment,c.core_international,c.tenure,c.credit_score,c.trds_mop_equal_2_or_gt_num,c.tot_amt_now_past_due_amt,b.bill1 ,b.bill2 ,b.bill3 ,b.bill4,b.bill5,
d.curr_bal,e.dish_restart,c.no_of_receivers

score$count_of_latecharge	<-	as.factor(	trim(score$count_of_latecharge)	)
score$total_equip_charge_amt <-	as.numeric(	score$total_equip_charge_amt	)
score$core_programming <- as.factor(score$core_programming)
score$all_star	<-	as.factor(	trim(score$all_star)	)
score$payment_method <- as.factor(score$payment_method)
score$line_of_business <- as.factor(score$line_of_business)
score$commitment <- as.factor(score$commitment)
score$core_international <- as.factor(score$core_international)
score$tenure	<-	as.numeric(	score$tenure	)
score$credit_score	<-	as.numeric(	score$credit_score	)
score$no_of_receivers <- as.factor(trim(score$no_of_receivers))
score$trds_mop_equal_2_or_gt_num	<-	as.numeric(	score$trds_mop_equal_2_or_gt_num	)
score$tot_amt_now_past_due_amt	<-	as.numeric(	score$tot_amt_now_past_due_amt	)
score$total_devices <- as.factor(score$total_devices)