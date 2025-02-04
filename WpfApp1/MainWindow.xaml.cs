using System.Diagnostics;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using Microsoft.ML.OnnxRuntimeGenAI;

namespace WpfApp1
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private Model? model = null;
        private Tokenizer? tokenizer = null;
        private readonly string ModelDir =
               AppDomain.CurrentDomain.BaseDirectory +
            @"Models\directml\directml-int4-awq-block-128";

        public MainWindow()
        {
            InitializeComponent();
            this.Activated += MainWindow_Activated;
        }

        private async void MainWindow_Activated(object? sender, EventArgs e)
        {
            if (model == null)
            {
                await InitializeModelAsync();
            }
        }

        public Task InitializeModelAsync()
        {
            Dispatcher.InvokeAsync(new Action(() =>
            {
                responseTextBlock.Text = "Loading model...";
            }));

            return Task.Run(() =>
            {
                var sw = Stopwatch.StartNew();
                model = new Model(ModelDir);
                tokenizer = new Tokenizer(model);
                sw.Stop();
                Dispatcher.InvokeAsync(new Action(() =>
                {
                    responseTextBlock.Text = $"Model loading took {sw.ElapsedMilliseconds} ms";
                }));
            });
        }

        public async IAsyncEnumerable<string> InferStreaming(string prompt)
        {
            if (model == null || tokenizer == null)
            {
                throw new InvalidOperationException("Model is not ready");
            }

            var generatorParams = new GeneratorParams(model);

            var sequences = tokenizer.Encode(prompt);

            generatorParams.SetSearchOption("max_length", 2048);
            generatorParams.SetInputSequences(sequences);
            generatorParams.TryGraphCaptureWithMaxBatchSize(1);

            using var tokenizerStream = tokenizer.CreateStream();
            using var generator = new Generator(model, generatorParams);
            StringBuilder stringBuilder = new();
            while (!generator.IsDone())
            {
                string part;
                try
                {
                    await Task.Delay(10).ConfigureAwait(false);
                    generator.ComputeLogits();
                    generator.GenerateNextToken();
                    part = tokenizerStream.Decode(generator.GetSequence(0)[^1]);
                    stringBuilder.Append(part);
                    if (stringBuilder.ToString().Contains("<|end|>")
                        || stringBuilder.ToString().Contains("<|user|>")
                        || stringBuilder.ToString().Contains("<|system|>"))
                    {
                        break;
                    }
                }
                catch (Exception ex)
                {
                    Debug.WriteLine(ex);
                    break;
                }

                yield return part;
            }
        }

        private async void myButton_Click(object sender, RoutedEventArgs e)
        {
            responseTextBlock.Text = "";

            if (model != null)
            {
                var systemPrompt = "You are a helpful assistant.";
                var userPrompt = promptTextBox.Text;

                var prompt = $@"<|system|>{systemPrompt}<|end|><|user|>{userPrompt}<|end|><|assistant|>";

                await foreach (var part in InferStreaming(prompt))
                {
                    responseTextBlock.Text += part;
                }
            }
        }
    }
}