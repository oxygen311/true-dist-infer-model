import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

new_method_name = "method from [2]"

if __name__ == "__main__":
    sns.set(style="whitegrid", font="serif", font_scale=1.2)
    plt.rc('text', usetex=True)

    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif', size=14)

    columns = ["n", "method", "relative error"]
    df = pd.DataFrame(columns=columns)

    our_method_df = pd.read_pickle("data/diff_n_errors_our_100_500(1000).pkl")
    tan_method_df = pd.read_pickle("data/diff_n_errors_tan_100_500(1000).pkl")

    tan_method_df["method"] = tan_method_df["method"].apply(lambda _: new_method_name)

    df = df.append(our_method_df, ignore_index=True)
    df = df.append(tan_method_df, ignore_index=True)

    print(df)
    df["absolute value of relative error"] = df["relative error"].apply(lambda x: abs(x))

    s = sns.boxplot(x="n", y="absolute value of relative error", hue="method", data=df, whis=[5, 95], showfliers=False,
                    palette="muted", linewidth=1.2)

    plt.legend(frameon=True)
    plt.subplots_adjust(bottom=0.11, top=0.98, right=0.99, left=0.1)

    plt.savefig('ns.pdf')
    plt.show()
