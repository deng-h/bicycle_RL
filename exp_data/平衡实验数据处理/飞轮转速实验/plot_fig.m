filename = '.\汇总.xlsx';
dataTable = readtable(filename, 'VariableNamingRule', 'preserve');

% 自定义数据的起始和终止位置
startIndex = 4000; % 起始行索引，从第 2 行开始（跳过标题行）
endIndex = 5000;   % 终止行索引

% 提取指定范围的数据
column1 = dataTable{startIndex:endIndex, 1}; % 第一列数据
column2 = dataTable{startIndex:endIndex, 2}; % 第二列数据

% 计算 column1 和 column2 的平均值
meanColumn1 = mean(column1);
meanColumn2 = mean(column2);

% 在 column1 和 column2 的数据上分别加上它们各自的平均值
column1 = column1 - meanColumn1;
column2 = column2 - meanColumn2;

figure;  % 创建一个新的图形窗口
% subplot(2, 1, 1); % 选择第一个子图
plot(startIndex:endIndex, column1, 'b-');
xlabel('时间步', 'FontSize', 15); % x 轴标签
ylabel('转速/rpm', 'FontSize', 15);
% legend('有转速惩罚函数', 'FontSize', 15); % 显示图例
ylim([-10, 10]); 

% subplot(2, 1, 2); % 选择第二个子图
% plot(startIndex:endIndex, column2, 'b-');
% xlabel('时间步', 'FontSize', 15); % x 轴标签
% ylabel('转速/rpm', 'FontSize', 15);
% legend('没有转速惩罚函数', 'FontSize', 15); % 显示图例
% ylim([-10, 10]); 

% legend('有转速惩罚函数', '没有转速惩罚函数', 'FontSize', 15);
% xlabel('时间步', 'FontSize', 15); % x 轴标签
% ylabel('转速/rpm', 'FontSize', 15);
% axis auto; 
print('myfigure','-dsvg');  % 导出为 SVG
