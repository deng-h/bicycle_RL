% 读取数据
filename = '.\3\0.monitor.csv';
data = readtable(filename);

% 根据 t 升序排序
data = sortrows(data, 't');

% 提取 r 值
r_values = data.r;

% 获取数据长度
n = length(r_values);

% 设置采样间隔
interval = 14; % 您可以调整此值以选择采样间隔

% 采样数据
sampled_indices = 1:interval:n;
sampled_r = r_values(sampled_indices);

% 生成横坐标
x_values = sampled_indices;

% 绘制折线图
figure('Position', [100, 100, 1200, 400]); % 调整图像宽度
plot(x_values, sampled_r, '-', 'LineWidth', 1.1);
xlabel('回合数/轮', 'FontSize', 20);
ylabel('奖励值', 'FontSize', 20);
% title('Line Plot of r Values with Sampling');
% grid on;

% 保存图像为文件（可选）
% saveas(gcf, 'line_plot_r_values.bmp');
print('myfigure','-dsvg');  % 导出为 SVG


