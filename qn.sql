CREATE DATABASE INTERN;
USE INTERN;

CREATE TABLE EMPLOYEE(
emp_id int,
emp_name varchar(255),
gender varchar(255),
dept varchar(255),
education varchar(255),
join_month varchar(255),
salary int
);
INSERT INTO EMPLOYEE(emp_id, emp_name, gender, dept, education, join_month, salary) VALUES
(1001, 'Ajay', 'M', 'Engineering', 'Doctoral', 'January', 25),
(1002, 'Babloo', 'M', 'Engineering', 'UG', 'February', 23),
(1003, 'Chhavi', 'F', 'HR', 'PG', 'March', 15),
(1004, 'Dheeraj', 'M', 'HR', 'UG', 'January', 12),
(1005, 'Evina', 'F', 'Marketing', 'UG', 'March', 16),
(1006, 'Fredy', 'M', 'Sales', 'UG', 'December', 10),
(1007, 'Garima', 'F', 'Sales', 'PG', 'March', 10),
(1008, 'Hans', 'M', 'Admin', 'PG', 'November', 8),
(1009, 'Ivanka', 'F', 'Admin', 'Intermediate', 'April', 7),
(1010, 'Jai', 'M', 'Peon', 'High School', 'December', 4);

-- question 1
SELECT dept
FROM EMPLOYEE
WHERE education != 'UG'
GROUP BY dept
HAVING SUM(salary) >= 20;

-- question 2
SELECT dept, SUM(salary) AS TotalSalary
FROM Employee
GROUP BY dept
HAVING TotalSalary >= 15
ORDER BY TotalSalary DESC;

-- question 3
SELECT dept
FROM Employee
GROUP BY dept
HAVING AVG(Salary) > 35;

-- question 4
SELECT emp_name
FROM EMPLOYEE
WHERE Salary BETWEEN 14 AND 24;

-- question 5
SELECT emp_name, salary + 5 AS TotalSalary
FROM Employee;