
--base sqoop table
create external table sbx_enterpriseanalytics.empty_box_score_base1(
ACCT_NO string,
CA_NUMBER string,
LAST_UPDATE_DATE string,
DISCO_TYPE string,
EQUIP_STATUS string,
EQUIP_CHARGE_AMT string,
MODEL_NO string,
CURR_BAL double,
AVG_BILL double,
COUNT_OF_LATE_CHARGE int)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
location '/user/enterpriseanalytics/sbx_enterpriseanalytics.db/empty_box_score_base1';

--restartflag sqoop table
create external table sbx_enterpriseanalytics.empty_box_score_restflag(
ACCT_NO string,
RESTART_FLAG int)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
location '/user/enterpriseanalytics/sbx_enterpriseanalytics.db/empty_box_score_restflag';

--bill sqoop table
create external table sbx_enterpriseanalytics.empty_box_score_bill(
ACCT_NO string,
bill1 string,
bill2 string,
bill3 string,
bill4 string,
bill5 string)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
location '/user/enterpriseanalytics/sbx_enterpriseanalytics.db/empty_box_score_bill';

create external table sbx_enterpriseanalytics.empty_box_score_test1(
ACCT_NO string,
DISCO_TYPE string,
EQUIP_STATUS string,
AVG_BILL double,
COUNT_OF_LATE_CHARGE int,
CURR_BAL double,
total_devices bigint,
EQUIP_CHARGE_AMT bigint)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
location '/user/enterpriseanalytics/sbx_enterpriseanalytics.db/empty_box_score_test1';

INSERT OVERWRITE TABLE sbx_enterpriseanalytics.empty_box_score_test1
select unique.ACCT_NO ,DISCO_TYPE ,EQUIP_STATUS ,AVG_BILL ,COUNT_OF_LATE_CHARGE ,CURR_BAL,total_devices ,EQUIP_CHARGE_AMT 
from
(select x.ACCT_NO ,x.DISCO_TYPE ,x.EQUIP_STATUS ,x.AVG_BILL ,x.COUNT_OF_LATE_CHARGE ,x.CURR_BAL 
from
(select ACCT_NO ,DISCO_TYPE ,EQUIP_STATUS ,AVG_BILL ,COUNT_OF_LATE_CHARGE,CURR_BAL, ROW_NUMBER () OVER (PARTITION BY ACCT_NO ORDER BY EQUIP_CHARGE_AMT DESC) AS RNK
from sbx_enterpriseanalytics.empty_box_score_base1
where disco_type IN('NP')
)x
where RNK =1) unique
join
(select ACCT_NO,count(distinct CA_NUMBER) as total_devices,sum(cast(EQUIP_CHARGE_AMT as int)) EQUIP_CHARGE_AMT
from sbx_enterpriseanalytics.empty_box_score_base1
where disco_type IN('NP')
group by ACCT_NO) aggr
on unique.ACCT_NO = aggr.ACCT_NO


create external table sbx_enterpriseanalytics.empty_box_score_re(
equip_status string,
acct_no string,
restart_flag int,
equip_charge_amt bigint,
core_programming string,
line_of_business string,
all_star string,
payment_method string,
count_of_late_charge int,
curr_bal double,
avg_bill double,
bnk_revlvng_trds_num string,
bill1 string,
bill2 string,
bill3 string,
bill4 string,
bill5 string,
prime_post_office_name string
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
location '/user/enterpriseanalytics/sbx_enterpriseanalytics.db/empty_box_score_re';

INSERT OVERWRITE TABLE sbx_enterpriseanalytics.empty_box_score_re
select distinct 
e.equip_status ,
e.acct_no ,
coalesce(r.restart_flag,0) ,
e.equip_charge_amt ,
c.core_programming ,
c.line_of_business ,
c.all_star ,
c.payment_method ,
e.count_of_late_charge ,
e.curr_bal ,
e.avg_bill ,
c.bnk_revlvng_trds_num ,
coalesce(b.bill1,0),
coalesce(b.bill2,0) ,
coalesce(b.bill3,0) ,
coalesce(b.bill4,0) ,
coalesce(b.bill5,0) ,
c.prime_post_office_name 
 from  
sbx_enterpriseanalytics.empty_box_score_test1 e 
join cust_aggr.cust_aggr c
ON e.acct_no=c.acct_num
LEFT JOIN sbx_enterpriseanalytics.empty_box_score_restflag r
ON e.acct_no=r.acct_no
LEFT JOIN sbx_enterpriseanalytics.empty_box_score_bill b
ON e.acct_no=b.acct_no
where e.disco_type='NP' 
