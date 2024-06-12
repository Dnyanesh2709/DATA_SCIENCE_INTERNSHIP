CREATE database COMPANY;
USE COMPANY;
CREATE TABLE EMPLOYEE(
emp_id int,
emp_name varchar(255),
emp_age int,
emp_bdate date
);
alter table EMPLOYEE
add column salary int;

RENAME TABLE EMPLOYEE TO employees;

INSERT INTO employees(
emp_id,emp_name,emp_age,emp_bdate,salary)
values (234,'rahul',24,'2024-04-04',200000);
SELECT * FROM employees;
INSERT INTO employees(
emp_id,emp_name,emp_age,emp_bdate,salary)
values (345,'dnyanesh',34,'2004-10-10',24000000);
SELECT * FROM employees where emp_id=345;
INSERT INTO employees(
emp_id,emp_name,emp_age,emp_bdate,salary)
values (344,'krithik',35,'2004-11-10',2400000);
INSERT INTO employees(
emp_id,emp_name,emp_age,emp_bdate,salary)
values (346,'Rohan',36,'2004-09-10',2400);

-- AGGREGATE FUNCTIONS

-- 1(COUNT)
SELECT COUNT(emp_name) AS employee_count
FROM employees;

-- 2(SUM)
 
SELECT SUM(salary) AS total_salary
FROM employees;

-- 3(AVG)
SELECT AVG(salary) AS average_salary
FROM employees;

-- 4(MAX)
SELECT MAX(salary) AS highest_salary
FROM employees;

-- 5(MIN)
SELECT MIN(salary) AS lowest_salary
FROM employees;

ALTER TABLE employees
ADD COLUMN gender int,
ADD COLUMN dept VARCHAR(255),
ADD COLUMN experience VARCHAR(255);

ALTER TABLE employees
MODIFY gender VARCHAR(255);

-- WHERE AND UPDATE
SET SQL_SAFE_UPDATES = 0;
UPDATE employees
SET gender = 'male', dept = 'AIDS', experience = '5 years'
WHERE emp_id = 234;

UPDATE employees
SET gender = 'male', dept = 'IT', experience = '5 years'
WHERE emp_id = 345;

UPDATE employees
SET gender = 'male', dept = 'EXTC', experience = '4 years'
WHERE emp_id = 344;

UPDATE employees
SET gender = 'male', dept = 'comp', experience = '5 years'
WHERE emp_id = 346;

ALTER TABLE employees
ADD COLUMN city VARCHAR (255);
UPDATE employees
SET city = 'MUMBAI'
WHERE emp_id=234;

UPDATE employees
SET city = 'DELHI'
WHERE emp_id=344;

UPDATE employees
SET city = 'CHENNAI'
WHERE emp_id=345;

UPDATE employees
SET city = 'GOA'
WHERE emp_id=346;

SELECT * FROM employees;

-- IN FUNCTION SYNTAX
-- SELECT COLUMN_NAME FROM TABLE_NAME
-- WHERE CONDITION

SELECT emp_name
FROM employees
WHERE city IN ('Mumbai', 'Delhi');
SELECT emp_name,emp_age,CITY from employees
where emp_age<40 and city='Mumbai';

SELECT emp_name from employees where NOT city='DELHI';


-- ORDER BY CLAUSE
SELECT emp_name,salary FROM employees
ORDER BY salary desc;

SELECT emp_name,salary FROM employees
ORDER BY salary asc;


UPDATE employees
SET city = 'CHENNAI'
WHERE emp_id=344;


-- GROUP BY CLAUSE
select city,count(emp_id) as total
from employees
group by city;

-- HAVING CLAUSE
SELECT city,sum(salary) as TOTAL_SALARY
from employees
group by city
having sum(salary)>= 2000;

-- LIKE OPERATOR
select city from employees 
where city LIKE 'M%';

